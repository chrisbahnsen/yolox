import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

print(tf.__version__)

onnx_model = onnx.load("firstwords.onnx")
onnx.checker.check_model(onnx_model)

tf_rep = prepare(onnx_model)
tf_rep.export_graph('model')

# Convert saved model to tflite
converter = tf.lite.TFLiteConverter.from_saved_model('model')
tf_lite_model = converter.convert()
open('firstwords.tflite','wb').write(tf_lite_model)

