

import cv2
import json
import onnxruntime as ort
import onnx
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

# Load image and bbox
image = cv2.imread("assets/colab/image_sportsfan.jpg")
with open('assets/colab/bbox_sportsfan.json') as jsonfile:
    bbox = json.load(jsonfile)['bbox']

dataset = 'wflw'
processor = SPIGAFramework(ModelConfig(dataset))
batch_crops, crop_bboxes = processor.pretreat(image, [bbox])

with open('spiga.onnx', 'rb') as f:  #see attached file
    model = onnx.load(f)

final_model = onnx.utils.polish_model(model)
onnx.save(final_model, 'spiga-p.onnx')
ort_sess = ort.InferenceSession('spiga-p.onnx')
onnx.checker.check_model(model, full_check=True)
import logging

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

# 在关键位置添加日志记录
logging.debug("Before creating InferenceSession")
ort_sess = ort.InferenceSession('spiga.onnx')
logging.debug("After creating InferenceSession")
outputs = ort_sess.run(None, {'input': crop_bboxes})

print("successfully")
