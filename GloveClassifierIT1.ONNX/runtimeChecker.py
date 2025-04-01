import onnxruntime as ort
import onnxruntime.backend
model_path = "model.onnx"

#https://microsoft.github.io/onnxruntime/
ort_sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider'])


print( ort.get_device()  )
