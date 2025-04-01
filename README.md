# Glove Detectors
This repo holds a few iterations and revisions of a model I built using the [Custom Vision](https://customvision.ai) portal from Microsoft.

* **GloveClassifierIT1.ONNX** - "Classifier" model, which just looks at the video frame to determine whether or not a glove is present, with confidence presented.
* **GloveDetectorIT1.ONNX** - "Detector 1" model, which draws a box around detected Glove features; performance is poor as input to this model was not consistent with operating environment.
* **GloveDetectorIT2.ONNX** - "Detector 2" model, which draws a box around detected Glove features as well as any present Defect features; better performance, since model inputs were taken from the same camera as the inferencing camera.

## Sample

Click sample video for better quality in youtube:

[![Sample video](/Glove_Detectors.gif)](https://www.youtube.com/watch?v=0a-5be_MPkY)

## How To 
In order to run the samples, download the code and model files, connect a camera via USB (validate that your device ID matches that indicated in the code), and run the command to start:

`./python3 predict.py ./model.onnx ./test1.jpg`