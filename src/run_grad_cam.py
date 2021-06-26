from grad_cam import gc
from model import initialize_model
import torch
import os
from shutil import rmtree

model_name = 'resnet'
num_classes = 2
feature_extract = False

model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model.load_state_dict(torch.load('./IH_resnet_weights.pt'))
data_path = './imgs_to_test_grad_cam/'
classes = ['noIH', 'IH']
output_dir = './results'
if os.path.exists(output_dir):
    rmtree(output_dir)
os.makedirs(output_dir)
gc(data_path, output_dir, model, classes)