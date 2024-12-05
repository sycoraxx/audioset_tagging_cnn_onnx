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
import onnxruntime
import onnx
import time

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def onnx_audio_tagging(args):
    # Arguments & parameters
    onnx_path = args.onnx_path
    audio_path = args.audio_path
    sample_rate = args.sample_rate
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    onnx.checker.check_model(onnx.load(onnx_path))

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    numpy_input = waveform.cpu().detach().numpy()
    print(f"Input length: {len(numpy_input)}")

    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Get input and output names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # print(input_name, output_name)

    onnxruntime_outputs = ort_session.run(None, {input_name: numpy_input})
    # print(len(onnxruntime_outputs))

    clipwise_output = onnxruntime_outputs[0].flatten()
    sorted_indexes = np.argsort(clipwise_output)[::-1]
    
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]]))

    print("Size of onnxruntime_outputs = {}".format(len(onnxruntime_outputs)))
    if len(onnxruntime_outputs) > 1:
        embedding = onnxruntime_outputs[1].flatten()
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels



def sound_event_detection(args):
    """Inference sound event detection result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    hop_size = args.hop_size
    model_type = args.model_type
    onnx_path = args.onnx_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
    """(time_steps, top_k)"""

    # Plot result    
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
        hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save sound event detection visualization to {}'.format(fig_path))

    return framewise_output, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000) 
    parser_at.add_argument('--onnx_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000) 
    parser_sed.add_argument('--onnx_path', type=str, required=True)
    parser_sed.add_argument('--audio_path', type=str, required=True)
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        start_time = time.time()
        onnx_audio_tagging(args)
        end_time = time.time()

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)

    else:
        raise Exception('Error argument!')
    
    print(f"Inference Time: {end_time - start_time:.6f} seconds")