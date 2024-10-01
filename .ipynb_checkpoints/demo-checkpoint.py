import os
import cv2
import argparse
import torch
import random
import logging
import math
import numpy as np
import subprocess

from trainer.trainer import run_trainer
from trainer.trainer_utils import load_model

from utils.parser import arg_parser
from utils.logger import set_logger
from utils.seed import set_seed

import dataset.vocoder_LSP_sptk as vocoder_LSP_sptk

from config.utils import load_config

def main(args, logger):

    for key in args.keys():
        logger.info(f'{key}: {args[key]}')

    set_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


    # for the given pretrained model, img shape: [84, 84]
    frame_H = 84
    frame_W = 84

    # prepare input audio

    # For the galaxy S22, the sampling frequency is 44.1kHz
    samplingFrequency = 44100 #16000
    frameLength = 1024 #1024
    frameShift = 512 #512 # for 83.28 frames, 481 is 
    order = args.data.order
    alpha = args.data.alpha
    stage = args.data.stage
    n_mgc = order + 1
    
    # load model
    _, model, _ = load_model(args, n_mgc, frame_H, frame_W, device)
    model.eval()
    mgc_lsp_coeff_fname = args.audio_fname.replace('.wav', '.mgclsp')
    
    if not os.path.isfile(mgc_lsp_coeff_fname) or args.reset_mgclsp:
        # process mgc_lsp
        mgc_lsp_coeff, lf0 = vocoder_LSP_sptk.encode(
            args.audio_fname[:-4],
            samplingFrequency,
            frameLength,
            frameShift,
            order,
            alpha,
            stage
        )
    else:
        mgc_lsp_coeff = np.fromfile(mgc_lsp_coeff_fname, dtype=np.float32).reshape(-1, n_mgc)

    print(f'Complete processing mgclsp')
        
    mgc_lsp_coeff = torch.from_numpy(mgc_lsp_coeff)

    new_mgc_lsp_coeffs = mgc_lsp_coeff.transpose(1,0).clone()
    for mgclsp_idx, mgc_lsp_coef in enumerate(new_mgc_lsp_coeffs):
        mgc_lsp_mean = mgc_lsp_coef.mean()
        mgc_lsp_std = mgc_lsp_coef.std()
        stand = (mgc_lsp_coef - mgc_lsp_mean) / mgc_lsp_std

        mgc_lsp_coeff[:,mgclsp_idx] = stand

    n_frames = mgc_lsp_coeff.shape[0]
    frame_length = frameShift / samplingFrequency # idk why, the sampling frequency should be doubled
    fps = 1 / frame_length
    
    print(f'total_nframes: {n_frames}')
    print(f'total fps: {fps:.4f}')
    print(f'total video length: {n_frames / fps:.4f}')
    
    # put into the model and process the results
    with torch.no_grad():
        pred = model(mgc_lsp_coeff.view(1, n_frames, n_mgc)).view(n_frames, frame_W, frame_H, 1).detach().numpy() * 255
        
    pred = pred.astype(np.uint8)

    print(f'Prediction complete')
    # save video
    # Define video parameters
    output_file = args.audio_fname.replace('.wav', '.avi')  # Output video file name
    #fps = 30  # Frames per second
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (you can use other codecs like 'XVID', 'MJPG', etc.)
    frame_size = (frame_W, frame_H)  # Frame size

    # Create VideoWriter object
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    # Convert and write frames to video
    for i in range(n_frames):
        # Convert single-channel grayscale to 3-channel BGR
        bgr_frame = cv2.cvtColor(pred[i], cv2.COLOR_GRAY2BGR)
        out.write(bgr_frame)

    # Release the VideoWriter object
    out.release()

    print(f"Embedding audio to video with ffmpeg")

    # Paths to your input files
    input_video = output_file
    input_audio = args.audio_fname
    output_video = input_video.replace('.avi', '_embed.avi')

    # Get the duration of the video
    #result = subprocess.run(['ffmpeg', '-i', input_video], stderr=subprocess.PIPE, universal_newlines=True)
    #duration_line = [x for x in result.stderr.split('\n') if "Duration" in x][0]
    #duration = duration_line.split(',')[0].split('Duration:')[1].strip()

    # Embed the audio, looping if necessary
    command = [
        'ffmpeg', '-stream_loop', '-1', '-i', input_audio, '-i', input_video,
        '-shortest', '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_video
    ]

    subprocess.run(command)

    print(f"Video saved as {output_video}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mri')
    parser.add_argument('--exp_name', type=str, default='exp0000')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--audio_fname', type=str)
    parser.add_argument('--reset_mgclsp', action='store_true')

    new_args = load_config(parser.parse_args())
    
    logger = set_logger(new_args)
    main(new_args, logger)
