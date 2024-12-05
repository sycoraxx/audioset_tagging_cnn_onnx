import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config
import torch.onnx

def convert_onnx(args):
    # Arguments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type # must
    checkpoint_path = args.checkpoint_path # must
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    model = model.to(device)

    model.eval()
    size = (1, 224000)

    dummy_tensor = torch.randn(size).to(device)


    onnx_file_path = os.path.splitext(checkpoint_path)[0] + ".onnx"

    torch.onnx.export(
    model,                     # Your PyTorch model
    dummy_tensor,               # Dummy input tensor
    onnx_file_path,            # Output file name
    export_params=True,        # Store parameters (weights) inside the model
    opset_version=15,          # ONNX opset version (11 is widely supported)
    do_constant_folding=True,  # Optimize constant folding for inference
    input_names=['input'],     # Input names
    output_names=['output'],   # Output names
    dynamic_axes={             # Enable variable-length axes (optional)
        'input': {0: 'batch_size',
                  1: 'time_steps'}, 
        'output': {0: 'batch_size'}
    }
)
print("ONNX model sucessfully created.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    # subparsers = parser.add_subparsers(dest='mode')

    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000) 
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    convert_onnx(args)