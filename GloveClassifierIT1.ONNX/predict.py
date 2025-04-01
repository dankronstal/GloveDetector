#check https://forums.developer.nvidia.com/t/cannot-install-onnxruntime-on-jetson-nano/229474 for onnxruntime-gpu 1.10
# also https://github.com/microsoft/onnxruntime/issues/7499 for model engine caching
# -> export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1


import time
preload = time.time()
import argparse
import pathlib
import numpy as np
import onnx
import onnxruntime
import PIL.Image
import cv2


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
    outputs = list(outputs.values())[0]
    for index, score in enumerate(outputs[0]):
        print(f"Label: {index}, score: {score:.5f}")

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
    #outputs = model.predict(args.image_filepath)
    #print_outputs(outputs)
    
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
        #predictions = {'model_output': np.array([[0.94169397, 0.95830595]], dtype=np.float32)}
    
        confidence_1, confidence_2 = predictions['model_output'][0]
        confidence_1_percent = confidence_1 * 100
        confidence_2_percent = confidence_2 * 100
        border_color = get_border_color(confidence_1_percent)
        opencvImage = cv2.copyMakeBorder(opencvImage, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
        shapes = np.zeros_like(opencvImage, np.uint8)
        #opencvImage = cv2.rectangle(opencvImage, (0, 600), (660, 385), border_color, -1)
        cv2.rectangle(shapes, (0, 600), (660, 385), border_color, -1)
        
        out = opencvImage.copy()
        alpha = 0.35
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(opencvImage, alpha, shapes, 1 - alpha, 0)[mask]

        #print(predictions)
        #fx, fy = im_pil.size
        #for k in predictions:
        #    if k['probability'] > 0.85:
        #        print(k['probability'])
        #        tl = (int(k['boundingBox']['left']*fx),int(k['boundingBox']['top']*fy))
        #        br = (int(k['boundingBox']['left']*fx+k['boundingBox']['width']*fx),int(k['boundingBox']['top']*fy+k['boundingBox']['height']*fy))

        #                opencvImage = cv2.rectangle(opencvImage,tl,br,(255, 0, 0),2)
        
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        out = cv2.putText(out, f"FPS: {fps:0>5.2f} -=]-[=- Glove: {confidence_1_percent:0>5.2f}%", (25,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow('Glove Classifier IT1', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() # destroys the window showing image.
            cam.release() #Closes video file or capturing device.
            break
        #break
        
    
    


if __name__ == '__main__':
    main()
