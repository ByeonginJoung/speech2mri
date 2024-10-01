import os
import cv2
import numpy as np
import torch
import librosa
import soundfile

import noisereduce as nr
import subprocess
from torch.utils.data import Dataset
from glob import glob
import scipy.io
import torchaudio
import torchaudio.transforms as T

import dataset.vocoder_LSP_sptk as vocoder_LSP_sptk
from dataset.data_augmentation import Compose, AddGaussianNoise, SpectralScaling, TimeStretchMGC, RandomErasing, TimeStretch

import moviepy.editor as mp

from tqdm import tqdm

from utils.voice_converter import make_tts_like, make_tts_like_ver2

class MRI(Dataset):
    def __init__(self,
                 args,
                 val=False
                 ):


        # vocoder params

        # how can we get frameshift?
        #
        # video framerate 83.28fps, video frames 2963
        # total audio samples: (if samplingFreq 20k) --> 20k * 2963 / 83.28 = 711400 samples
        # to get a frames Shift,
        # (711400 frames - 1024 frame shift) / (2963 total frames - 1) ~= 240

        # for all cases, the corresponds audio samples to a single video frame is 534. this value is fixed for all audio
        # maybe similar to frameShift value

        # fps control ratio reducing the total frame and the model can get a more longer audio features to predict each frame
        # this value can make reliable results, since it took a more longer data, however, it can cause the problem of data leakage
        
        self.n_sample_audio = 534
        
        self.args = args
        self.samplingFrequency = args.data.samplingFrequency #20000
        self.fps_control_ratio = args.data.fps_control_ratio
        #self.frameLength = int(args.data.frameLength * args.data.fps_control_ratio)
        #self.frameShift = int(args.data.frameShift * args.data.fps_control_ratio) #481
        self.frameLength = args.data.frameLength 
        self.frameShift = args.data.frameShift
        self.order = args.data.order
        self.alpha = args.data.alpha
        self.stage = args.data.stage
        self.val = val
        self.cut_rand_initial = args.data.cut_rand_initial
        # get load random video

        shortest_video_nframes = 1e10

        if args.dataset_type == '75-speaker':
            video_fnames = sorted(glob(os.path.join(args.dataset_dir, f'sub{int(args.sub_name):03d}', '2drt', 'video', '*.mp4')))
            video_extension = '.mp4'
        elif args.dataset_type == 'timit':
            video_fnames = sorted(glob(os.path.join(args.dataset_dir, f'{args.sub_name}', 'avi_withaudio', '*.avi')))
            video_extension = '.avi'
                                  
            
        self.load_video(video_fnames[0])

        if args.data.feature_mode == 'mgclsp':
            self.n_input_feature = self.order + 1
            self.video_list = list()
            self.mgclsp_list = list()        
            for video_idx, video_fname in enumerate(video_fnames):
                if 'postures' in video_fname:
                    continue
                else:
                    if not val:
                        print(f'Processing video: {video_fname}')
                    audio_fname = video_fname.replace(video_extension, '.wav')

                    if not os.path.isfile(audio_fname):      
                        self.extract_audio_from_video(video_fname, audio_fname)

                    # extract de-noised audio    
                    converted_audio_fname = audio_fname.replace('.wav', '_convert.wav')
                    if not os.path.isfile(converted_audio_fname):
                        make_tts_like(audio_fname, converted_audio_fname)

                    #process = ['sox', '--i', converted_audio_fname]
                    #subprocess.run(process)

                    add_aud_fname = converted_audio_fname.replace('.wav', '.mgclsp')

                    # for debug
                    if os.path.isfile(add_aud_fname):
                        try:
                            mgc_lsp_coeff = self.load_mgc(add_aud_fname)
                            lf0 = self.load_lf0(add_aud_fname.replace('.mgclsp', '.lf0'))
                        except:
                            if not val:
                                print(f'data for fname: {video_fname} is invalid in first condition.')
                            continue
                    else:
                        mgc_lsp_coeff, lf0 = self.get_mgc_lsp_coeff(converted_audio_fname[:-4])

                    valid_mgc = mgc_lsp_coeff.reshape(-1).shape[0] % (self.order + 1) == 0
                    #valid_lf0 = lf0.shape[0] == mgc_lsp_coeff.shape[0]

                    if valid_mgc: # and valid_lf0:
                        self.video_list.append(video_fname)
                        self.mgclsp_list.append(mgc_lsp_coeff)

                        if not val:
                            video = self.load_video(video_fname)
                            diff_len = video.shape[-1] - mgc_lsp_coeff.shape[0]

                            print(f'len of vid: {video.shape[-1]} | len of mgc: {mgc_lsp_coeff.shape[0]} | diff len: {diff_len}')
                        """
                        video = self.load_video(video_fname)
                        if video.shape[-1] <= shortest_video_nframes:
                            shortest_video_nframes = video.shape[-1]
                            print(f'shortest_video_nframes are updated to: {shortest_video_nframes}')
                        """
                    else:
                        if not val:
                            print(f'data for fname: {video_fname} is invalid in second condition.')
                        continue

            if val:
                self.video_list = [self.video_list[12]]
                self.mgclsp_list = [self.mgclsp_list[12]]

            # stack all mgc lsp to calculate std and mean for the data
            if not val:
                all_mgclsp = np.concatenate(self.mgclsp_list, axis=0)

                self.mgc_mean = np.mean(all_mgclsp, axis=0)
                self.mgc_std = np.std(all_mgclsp, axis=0)
            else:
                self.mgc_mean = np.zeros(self.n_mgc)
                self.mgc_std = np.zeros(self.n_mgc)

        elif args.data.feature_mode == 'raw' or args.data.feature_mode == 'melspectogram':
            self.n_input_feature = self.n_sample_audio
            self.video_list = list()
            self.audio_list = list()
            
            for video_idx, video_fname in enumerate(video_fnames):
                if 'postures' in video_fname:
                    continue
                else:
                    if not val:
                        print(f'Processing video: {video_fname}')
                    audio_fname = video_fname.replace(video_extension, '.wav')

                    if not os.path.isfile(audio_fname):      
                        self.extract_audio_from_video(video_fname, audio_fname)

                    # extract de-noised audio    
                    converted_audio_fname = audio_fname.replace('.wav', '_convert.wav')
                    if not os.path.isfile(converted_audio_fname):
                        make_tts_like(audio_fname, converted_audio_fname)

                    converted_audio_fname_ver2 = audio_fname.replace('.wav', '_convert_ver2.wav')
                    if not os.path.isfile(converted_audio_fname_ver2):
                        make_tts_like_ver2(audio_fname, converted_audio_fname_ver2)
                        
                    self.video_list.append(video_fname)
                    self.audio_list.append(converted_audio_fname_ver2)

            if val:
                self.video_list = [self.video_list[12]]
                self.audio_list = [self.audio_list[12]]

            if args.data.feature_mode == 'melspectogram':
                self.mel_spectrogram = T.MelSpectrogram(
                    sample_rate=self.samplingFrequency,  # Sampling rate of the audio
                    n_fft=self.frameLength,         # Frame length
                    hop_length=self.frameShift,     # Frame shift
                    n_mels=args.model.in_feat           # Number of Mel bands
                )
                self.to_db = T.AmplitudeToDB()
            self.mgc_mean = 0
            self.mgc_std = 0
        else:
            raise NotImplementedError

        if args.data.augmentation.use_augment:
            aug_list = list()
            if args.data.augmentation.add_gaussian:
                aug_list.append(
                    AddGaussianNoise(
                        args.data.augmentation.add_gaussian_mean,
                        args.data.augmentation.add_gaussian_std)
                )
            if args.data.augmentation.add_spectral_scaling:
                aug_list.append(
                    SpectralScaling(
                        args.data.augmentation.spectral_scaling_factor)
                )
            if args.data.augmentation.add_time_stretch_mgc:
                aug_list.append(
                    TimeStretchMGC(
                        args.data.augmentation.time_stretch_factor)
                )
            if args.data.augmentation.add_random_erasing:
                aug_list.append(
                    RandomErasing(
                        args.data.augmentation.random_erasing_prob,
                        args.data.augmentation.random_erasing_size)
                )
            if args.data.augmentation.add_time_stretch:
                aug_list.append(
                    TimeStretch(
                        [args.data.augmentation.time_stretch_factor_min,
                         args.data.augmentation.time_stretch_factor_max])
                )
            self.augmentation = Compose(aug_list)
                            
    def __len__(self):
        return len(self.video_list)
    
    # load vocoder features,
    # or calculate, if they are not available
    def get_mgc_lsp_coeff(self, audio_fname):
        return vocoder_LSP_sptk.encode(
            audio_fname,
            self.samplingFrequency,
            self.frameLength,
            self.frameShift,
            self.order,
            self.alpha,
            self.stage)
    
    def extract_audio_from_video(self, video_path, audio_output_path):
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path)
    
    # from LipReading with slight modifications
    # https://github.com/hassanhub/LipReading/blob/master/codes/data_integration.py
    ################## VIDEO INPUT ##################
    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        fps = cap.get(cv2.CAP_PROP_FPS)

        #framesPerSec = 23.18
        
        # make sure that all the videos are the same FPS
        #if (np.abs(fps - framesPerSec) > 0.01):
        #    print('fps:', fps, '(' + video_path + ')')
        #    raise
        
        buf = np.empty((frameHeight, frameWidth, frameCount), np.dtype('float32'))
        fc = 0
        ret = True

        while (fc < frameCount  and ret):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype('float32')
            # min-max scaling to [0-1]
            frame = frame-np.amin(frame)
            # make sure not to divide by zero
            if np.amax(frame) != 0:
                frame = frame/np.amax(frame)
            buf[:,:,fc]=frame
            fc += 1
        cap.release()

        self.frameHeight = frameHeight
        self.frameWidth = frameWidth
        self.fps = fps
        #print(f'video_fname: {video_path} | fps: {fps}')
        
        return buf

    def load_mgc(self, fname):
        return np.fromfile(fname, dtype=np.float32).reshape(-1, self.order + 1)

    def load_lf0(self, fname):
        return np.fromfile(fname, dtype=np.float32)

    def __getitem__(self, idx):

        # load video data here
        if self.args.data.feature_mode == 'mgclsp':
            video_fname = self.video_list[idx]
            mgc_fname = video_fname.replace(video_extension, '_convert.mgclsp')

            video = self.load_video(video_fname)
            mgc_lsp_coeff = self.load_mgc(mgc_fname)

            vd_length = min(video.shape[-1], mgc_lsp_coeff.shape[0])

            video = video[:,:,:vd_length]
            mgc_lsp_coeff = mgc_lsp_coeff[:vd_length,:]

            video = torch.from_numpy(video).permute(2,0,1)
            mgc_lsp_coeff = torch.from_numpy(mgc_lsp_coeff)

            # we need to cut them randomly to sample initial things
            if self.cut_rand_initial:
                rand_len = np.random.randint(200)

                video = video[rand_len:]
                mgc_lsp_coeff = mgc_lsp_coeff[rand_len:]

            if not self.val:
                mgc_lsp_coeff, video = self.augmentation(mgc_lsp_coeff, video)

            mgc_lsp_coeff = (mgc_lsp_coeff - self.mgc_mean[None,...]) / self.mgc_std[None,...]

            return video, mgc_lsp_coeff, video_fname
        
        elif self.args.data.feature_mode == 'raw':
            video_fname = self.video_list[idx]
            audio_fname = self.audio_list[idx]
            # Load the audio file
            waveform, sr = torchaudio.load(audio_fname)

            # Resample if necessary
            if sr != self.samplingFrequency:
                resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.samplingFrequency)
                waveform = resample_transform(waveform)

            # Load the target MRI image (assuming they are preprocessed and stored as tensors)
            video = self.load_video(video_fname)
            video = torch.from_numpy(video).permute(2,0,1)
               
            # cut the audio
            len_vid = video.shape[0]
            video = video.squeeze()
            if self.n_sample_audio * len_vid > audio.shape[0]:
                max_len = int(np.floor(audio.shape[0] / self.n_sample_audio))
                audio = audio[:int(self.n_sample_audio * max_len)].squeeze()
                audio = audio.view(max_len, -1)
                video = video[:max_len]
            else:
                audio = audio[:self.n_sample_audio * len_vid].squeeze()
                audio = audio.view(len_vid, -1)
                
            if not self.val:
                audio, video = self.augmentation(waveform.T, video.unsqueeze(-1))
            else:
                audio = waveform.T

            audio_min = audio.min()
            audio_max = audio.max()

            # Normalize the waveform (optional)
            #waveform = (waveform - waveform.mean()) / waveform.std()
            audio = (audio - audio_min) / (audio_max - audio_min)
                
            return video, audio, video_fname
        elif self.args.data.feature_mode == 'melspectogram':
            video_fname = self.video_list[idx]
            audio_fname = self.audio_list[idx]
            
            audio, sr = torchaudio.load(audio_fname)
            mel_spec = self.mel_spectrogram(audio)
            mel_spec_db = self.to_db(mel_spec).squeeze()
            
            video = self.load_video(video_fname)
            video = torch.from_numpy(video).permute(2,0,1)

            vd_length = min(video.shape[0], mel_spec_db.shape[-1])

            video = video[:vd_length,:,:]
            mel_spec_db = mel_spec_db[:,:vd_length].T

            # before make augmentation, make sure that all input's shape has a length at first
            if not self.val:
                audio, video = self.augmentation(mel_spec_db, video.unsqueeze(-1))
            else:
                audio = mel_spec_db

            audio_min = -80
            audio_max = 20

            audio = (audio - audio_min) / (audio_max - audio_min)

            video = video[::int(self.fps_control_ratio)].squeeze()
            #audio = audio[::int(self.fps_control_ratio)].squeeze()

            return video.squeeze(), audio, video_fname
            
        else:
            raise NotImplementedError
