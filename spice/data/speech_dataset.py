from __future__ import print_function
from PIL import Image
import os.path
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch, torchaudio, torchvision
import librosa
SR=16000
FRAME_LEN=5 #seconds

class SpeechDataset(Dataset):
    splits = ('train','test')
    def __init__(self, split='train',feature_type='spectrogram'):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        # self.transform1 = None #transform1 
        # self.transform2 = None #transform2
        self.split = split  # train/test
        self.data = None
        self.feature_type = feature_type
        if self.feature_type == 'spectrogram':
            self.transform = torchaudio.transforms.Spectrogram(n_fft=512,
                                                               hop_length=512//4,
                                                               normalized=True,
                                                               center=True,
                                                               pad_mode="reflect",
                                                               power=2.0,
                                                               onesided=True)
        elif self.feature_type == 'mel': 
            self.transform = torchaudio.transforms.MelSpectrogram(n_fft=512,
                                                                    sample_rate=SR,
                                                                    hop_length=512//4,
                                                                    normalized=True,
                                                                    center=True,
                                                                    pad_mode="reflect",
                                                                    power=2.0,
                                                                    norm="slaney",
                                                                    onesided=True,
                                                                    n_mels=128,
                                                                    mel_scale="htk")
        else: #MFCC 
            self.feature_type='MFCC'
            self.transform = torchaudio.transforms.MFCC(n_mfcc=128,
                                                        sample_rate=SR,melkwargs={"n_fft": 512,"n_mels": 128,"hop_length": 512//4,"mel_scale": "htk"},)
        self.transform_channels = torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        self.__loadfile(self.split)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
      
        path,target = self.data.iloc[index].full_path,self.data.iloc[index].label
        wav,fs = torchaudio.load(path)
        wav = wav/torch.max(abs(wav)) #norm
        if fs != SR:
            wav = torchaudio.transforms.Resample(orig_freq=fs,new_freq=SR)(wav)
        wav=wav.flatten()
        env_start,env_stop = self.amplitude_envelope_start_stop(wav)
        if env_stop-env_start < FRAME_LEN:
            wav = torch.cat((wav,torch.zeros(2*FRAME_LEN*SR)))

        wav = wav[int(env_start*SR):int((env_start)*SR+FRAME_LEN*SR)]
        if self.feature_type!='MFCC':
            feature_image= 20*torch.log10(self.transform(wav).unsqueeze(0)+1e-10)
        else:
            feature_image = self.transform(wav).unsqueeze(0)
        # feature_image= self.transform(wav).unsqueeze(0)
        feature_image_gs = self.transform_channels(feature_image)
        return feature_image_gs, target, index

    def __len__(self):
        return int(len(self.data))

    def __loadfile(self, split): #change to get labels from speakers name/folder name
        data_file=f'/dsi/gannot-lab1/LibriSpeech_mls_french/{split}/transcripts.txt'
        df = pd.read_table(data_file,names=['file_name','transcript'])
        path_to_data = f'/dsi/gannot-lab1/LibriSpeech_mls_french/{split.capitalize()}'
        self.data = pd.DataFrame(columns=["full_path","label"])
        self.data.full_path = df['file_name'].apply(lambda x: self.mls_filename_to_path(path_to_data,x))
        self.data.label = df['file_name'].apply(lambda x: x.split('_')[0]).to_list()


    def mls_filename_to_path(self,path,name):
        if name == None:
            return None
        path = Path(path)
        name = name.split('_')
        return str((path/name[0]/name[1]/f'{name[0]}_{name[1]}_{name[2]}.wav'))
        
    def amplitude_envelope_start_stop(self,signal, frame_size=512, hop_length=256,sr=16000):
        """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
        amplitude_envelope = []
        
        # calculate amplitude envelope for each frame
        for i in range(0, len(signal), hop_length): 
            amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
            amplitude_envelope.append(amplitude_envelope_current_frame)
        env= np.array(amplitude_envelope)
        t = librosa.frames_to_time(range(len(env)),sr = sr,hop_length=hop_length)
        start = 0
        stop = len(env)-1
        for i in range(len(env)):
            if env[i]>0.1:
                start = i
                break
        for i in range(len(env),0):
            if env[i]>0.1:
                stop = i
                break
        return t[start],t[stop]