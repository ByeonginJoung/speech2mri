import torch
import torch.nn.functional as F
import numpy as np
import librosa

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x, video):
        for transform in self.transforms:
            x, video = transform(x, video)
        return x, video

class TimeStretch:
    def __init__(self, stretch_factor=[0.9, 1.1]):
        self.stretch_factor = stretch_factor

    def __call__(self, audio, video):
        stretch_factor = np.random.uniform(self.stretch_factor[0], self.stretch_factor[1])
        
        # Audio Stretching
        L_audio, C_audio = audio.shape
        stretched_length_audio = int(L_audio * stretch_factor)
        audio = F.interpolate(audio.unsqueeze(0).unsqueeze(0), size=(stretched_length_audio, C_audio), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
        # Video Stretching
        L_video, H, W, C_video = video.shape
        stretched_length_video = int(L_video * stretch_factor)
        video = F.interpolate(video.permute(3, 0, 1, 2).unsqueeze(0), size=(stretched_length_video, H, W), mode='trilinear', align_corners=False).squeeze(0).permute(1, 2, 3, 0)
        
        return audio, video
    
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, x, video):
        #noise = np.random.randn(*x.shape) * self.std + self.mean
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise, video
    
class SpectralScaling:
    def __init__(self, scale_factor=0.1):
        self.scale_factor = scale_factor
    
    def __call__(self, x, video):
        return x * (1 + self.scale_factor * torch.randn_like(x)), video

class TimeStretchMGC:
    def __init__(self, stretch_factor=1.1):
        self.stretch_factor = stretch_factor
    
    def __call__(self, audio, video):
        L_audio, C_audio = audio.shape
        L_video, H, W = video.shape
        
        stretched_length_audio = int(L_audio * self.stretch_factor)
        stretched_length_video = int(L_video * self.stretch_factor)
        
        # Stretch audio
        stretched_audio = F.interpolate(
            audio.unsqueeze(0).unsqueeze(0),
            size=(stretched_length_audio, C_audio),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        
        # Stretch video
        stretched_video = F.interpolate(
            video.unsqueeze(0).unsqueeze(0),  # Change to (C, L, H, W)
            size=(stretched_length_video, H, W),
            mode='trilinear',
            align_corners=False
        ).squeeze()  # Back to (L, H, W, C)
        
        return stretched_audio, stretched_video
    
class RandomErasing:
    def __init__(self, erase_prob=0.5, erase_size=0.1):
        self.erase_prob = erase_prob
        self.erase_size = erase_size
    
    def __call__(self, audio, video):
        # Assuming video shape is (Frames, H, W, C)
        if torch.rand(1).item() > self.erase_prob:
            return audio, video
        
        num_erase_audio = int(audio.size(0) * self.erase_size)
        num_erase_video = int(video.size(0) * self.erase_size)
        start_audio = torch.randint(0, audio.size(0) - num_erase_audio, (1,)).item()
        start_video = torch.randint(0, video.size(0) - num_erase_video, (1,)).item()
        
        audio[start_audio:start_audio+num_erase_audio, :] = 0
        video[start_video:start_video+num_erase_video, :, :] = 0
        
        return audio, video
