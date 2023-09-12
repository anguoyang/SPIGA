

import cv2
import json
import numpy as np
import torch

# Load image and bbox
image = cv2.imread("assets/colab/image_sportsfan.jpg")
with open('assets/colab/bbox_sportsfan.json') as jsonfile:
    bbox = json.load(jsonfile)['bbox']

from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

# Process image
dataset = 'wflw'
processor = SPIGAFramework(ModelConfig(dataset))

#batch_crops, crop_bboxes = processor.pretreat(image, [bbox])
#outputs = self.net_forward(batch_crops)
#features = self.postreatment(outputs, crop_bboxes, bboxes)
features = processor.inference(image, [bbox])

"""
##############################################################
net = processor.model
net.eval()
print('Finished loading model!')
print(net)
device = torch.device("cpu")
net = net.to(device)

##################export###############
output_onnx = 'faceDetector.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
#input_names = ["input0"]
#output_names = ["output0"]
#inputs = batch_crops
torch.onnx.export(net, batch_crops, "spiga.onnx", opset_version = 18)  # Use opset_version 11
#torch.onnx.dynamo_export(net, batch_crops).save("./b.onnx")

#torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
#                                   input_names=input_names, output_names=output_names)
##################end###############

"""
import copy
from spiga.demo.visualize.plotter import Plotter

# Prepare variables
x0,y0,w,h = bbox
canvas = copy.deepcopy(image)
landmarks = np.array(features['landmarks'][0])
headpose = np.array(features['headpose'][0])

# Plot features
plotter = Plotter()
canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
canvas = plotter.hpose.draw_headpose(canvas, [x0,y0,x0+w,y0+h], headpose[:3], headpose[3:], euler=True)

# Show image results
(h, w) = canvas.shape[:2]
canvas = cv2.resize(canvas, (512, int(h*512/w)))
#cv2.imshow("spiga",canvas)
cv2.imwrite("spiga.png", canvas)

"""
##############################################################
net = processor.model
net.eval()
print('Finished loading model!')
print(net)
device = torch.device("cpu")
net = net.to(device)

##################export###############
output_onnx = 'faceDetector.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input0"]
output_names = ["output0"]
inputs = [torch.randn(1, 3, 256, 256).to(device), processor.model3d, processor.cam_matrix]
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names)
##################end###############

print("successfully")
"""






