# if model is broken, try running symbolic_shape_infer.py against it first...
import time
preload = time.time()
import argparse
import pathlib
import numpy as np
import onnx
import onnxruntime
import PIL.Image
import cv2

PROB_THRESHOLD = 0.5  # Minimum probably to show results.


class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}
        
        
    def predict_cam(self, img):
        image = img.resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}


def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            print(f"Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")


def get_border_color(confidence):
    if confidence < 75:
        # Solid red for confidence below 75%
        return (0, 0, 255)
    elif 75 <= confidence <= 100:
        # Interpolating between red and green for confidence between 75% and 100%
        # 75% confidence -> red (255, 0, 0)
        # 87.5% confidence -> yellow (255, 255, 0)
        # 100% confidence -> green (0, 255, 0)
        
        normalized_conf = (confidence - 75) / 25  # Normalize to range [0, 1]
        red = int(255 * (1 - normalized_conf))
        green = int(255 * normalized_conf)
        return (0, green, red)
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filepath', type=pathlib.Path)
    parser.add_argument('image_filepath', type=pathlib.Path)

    args = parser.parse_args()

    model = Model(args.model_filepath)
    loadtime = time.time()-preload
    print(f"load time: {loadtime}")
    
    class_id_labels = ["Defect", "Glove"]
    class_id_colors = [(0, 0, 255), (0, 255, 0)]
    
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        start_time = time.time()        
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = PIL.Image.fromarray(img)
        opencvImage = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        predictions = model.predict_cam(im_pil)

        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        
        # Get image dimensions
        height, width = opencvImage.shape[:2]

        # Iterate through detected objects
        for box, class_id, score in zip(predictions['detected_boxes'][0], predictions['detected_classes'][0], predictions['detected_scores'][0]):
            if score > PROB_THRESHOLD:
                # Convert normalized coordinates to pixel values
                x_min = int(box[0] * width)
                y_min = int(box[1] * height)
                x_max = int(box[2] * width)
                y_max = int(box[3] * height)
                
                border_color = class_id_colors[class_id]
                label = class_id_labels[class_id]
                score = score * 100
                
                # Draw the bounding box
                opencvImage = cv2.rectangle(opencvImage, (x_min, y_min), (x_max, y_max), border_color, 2)
                
                # Display the class label and confidence score
                label = f"{label}: {score:0>5.2f}%"
                cv2.putText(opencvImage, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)


        opencvImage = cv2.copyMakeBorder(opencvImage, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        shapes = np.zeros_like(opencvImage, np.uint8)
        
        cv2.rectangle(shapes, (0, 600), (660, 385), (255, 0, 0), -1)
        
        out = opencvImage.copy()
        alpha = 0.35
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(opencvImage, alpha, shapes, 1 - alpha, 0)[mask]
        
        out = cv2.putText(out, f"FPS: {fps:0>5.2f}", (25,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow('Glove Detector IT2', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() # destroys the window showing image.
            cam.release() #Closes video file or capturing device.
            break
            
if __name__ == '__main__':
    main()
