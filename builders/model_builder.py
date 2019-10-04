import sys, os
import tensorflow as tf
import subprocess

sys.path.append("models")
from models.FRRN import build_frrn


def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])



def build_model(model_name, net_input, num_classes, crop_width, crop_height, is_training=True):
	# Get the selected model. 
	# Some of them require pre-trained ResNet

	print("Preparing the model ...")

	network = None
	init_fn = None

	network = build_frrn(net_input, preset_model = model_name, num_classes=num_classes)


	return network, init_fn