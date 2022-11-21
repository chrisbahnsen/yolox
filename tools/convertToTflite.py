import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
import argparse


def convertToTflite(onnxPath):
    

    onnx_model = onnx.load(onnxPath)
    onnx.checker.check_model(onnx_model)

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('model')

    # Convert saved model to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model('model')
    tf_lite_model = converter.convert()

    tflitepath = onnxPath.replace('.onnx', '.tflite')
    open(tflitepath,'wb').write(tf_lite_model)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Utility script for converting a ONNX model to tflite")
    parser.add_argument('--modelPath', type=str, default="animal.onnx", help="File name of the ONNX model to convert")

    args = parser.parse_args()

    convertToTflite(args.modelPath)
