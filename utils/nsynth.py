"""
File: nsynth.py
Author: Kwon-Young Choi
Email: kwon-young.choi@hotmail.fr
Date: 2018-11-13
Description: Load NSynth dataset using pytorch Dataset.
If you want to modify the output of the dataset, use the transform
and target_transform callbacks as ususal.
"""
import os
import json
import glob
import numpy as np
import scipy.io.wavfile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchaudio
import sys
from sklearn.preprocessing import LabelEncoder


class NSynth(data.Dataset):

    """Pytorch dataset for NSynth dataset
    args:
        root: root dir containing examples.json and audio directory with
            wav files.
        transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        blacklist_pattern: list of string used to blacklist dataset element.
            If one of the string is present in the audio filename, this sample
            together with its metadata is removed from the dataset.
        categorical_field_list: list of string. Each string is a key like
            instrument_family that will be used as a classification target.
            Each field value will be encoding as an integer using sklearn
            LabelEncoder.
    """

    def __init__(self, root, transform=None, target_transform=None,
                 blacklist_pattern=[],
                 categorical_field_list=["instrument_family"]):
        """Constructor"""
        assert(isinstance(root, str))
        assert(isinstance(blacklist_pattern, list))
        assert(isinstance(categorical_field_list, list))
        self.root = root
        self.filenames = glob.glob(os.path.join(root, "audio/*.wav"))
        with open(os.path.join(root, "examples.json"), "r") as f:
            self.json_data = json.load(f)
        for pattern in blacklist_pattern:
            self.filenames, self.json_data = self.blacklist(
                self.filenames, self.json_data, pattern)
        self.categorical_field_list = categorical_field_list
        self.le = []
        for i, field in enumerate(self.categorical_field_list):
            self.le.append(LabelEncoder())
            field_values = [value[field] for value in self.json_data.values()]
            self.le[i].fit(field_values)
        self.transform = transform
        self.target_transform = target_transform

    def blacklist(self, filenames, json_data, pattern):
        filenames = [filename for filename in filenames
                     if pattern not in filename]
        json_data = {
            key: value for key, value in json_data.items()
            if pattern not in key
        }
        return filenames, json_data

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (audio sample, *categorical targets, json_data)
        """
        name = self.filenames[index]
        _, sample = scipy.io.wavfile.read(name)
        target = self.json_data[os.path.splitext(os.path.basename(name))[0]]
        categorical_target = [
            le.transform([target[field]])[0]
            for field, le in zip(self.categorical_field_list, self.le)]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return [sample, *categorical_target, target]

    def getAudioCharacteristics(self, index):
        item = self.__getitem__(index)
        return {"fichero": item[3]["note_str"]+".wav","familia":item[3]["instrument_source"]}

    # def getSignal(self,index):
    #     #La ruta va desde el fichero donde se encuentra el jupyter notebook
    #     return torchaudio.load("nsynth-valid/audio/"+self.getAudioCharacteristics(index)["fichero"])

    # def getLabel(self,index):
    #     return self.getAudioCharacteristics(index)["familia"]

    # def samplingTransform(self,signal,sample_rate):
    #     #Usamos sample_rate 16000 por defecto
    #     if sample_rate != self.TARGET_SAMPLE_RATE:
    #         signal = torchaudio.transforms.signalResample(sample_rate,self.TARGET_SAMPLE_RATE)(signal)
    #     return signal
            

    # def monoTransform(self,signal):
    #     if signal.shape[0] > 1:
    #         signal = torch.mean(signal, dim=0,keepdim=True)
    #     return signal
    
    # def setTargetSampleRate(self,sample_rate):
    #     self.TARGET_SAMPLE_RATE=sample_rate
    #     return 0

    # def setTransformation(self,transformation):
    #     self.TRANSFORMATION=transformation
    #     return 0

class SignalTransformation():

     #Sample rate por defecto
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TARGET_SAMPLE_RATE=16000
    #Usamos espectogramas de MEL por defecto al utilizar instrumentos musicales para captar las frecuencias melódicas de forma entendible para humanos y para el modelo
    TRANSFORMATION=torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE,
                                                      n_fft=1024, 
                                                      hop_length=512,
                                                      n_mels=64).to(device)

    def __init__(self, fichero, label):
        self.fichero=fichero
        self.label=label
    
    def generarSpectrograma(self):
        signal = self.TRANSFORMATION(self.getSignalTuned())
        return signal

    @classmethod
    def generarSpectrogramaFromSignal(cls, signal):
        signal = cls.monoTransformClass(signal)
        signal = cls.samplingTransformClass(signal,cls.TARGET_SAMPLE_RATE)
        signal = cls.TRANSFORMATION(signal)
        return signal

    @classmethod
    def generarSTFTFromSignal(cls,signal):
        signal = cls.monoTransformClass(signal)
        signal = cls.samplingTransformClass(signal,cls.TARGET_SAMPLE_RATE)
        signal = torchaudio.transforms.Spectrogram().to(cls.device)(signal)
        return signal
    def getSignal(self):
        #La ruta va desde el fichero donde se encuentra el jupyter notebook
        return torchaudio.load("nsynth-valid/audio/"+self.fichero)

    def getSignalTuned(self):
        signal, sr  = self.getSignal()
        signal= self.monoTransform(signal)
        signal = self.samplingTransform(signal,self.TARGET_SAMPLE_RATE)
        return signal 

    def getLabel(self):
        return self.label
    def getSignalAndLabel(self):
        return self.getSignal(), self.getLabel()

    def samplingTransform(self,signal,sample_rate):
        #Usamos sample_rate 16000 por defecto
        if sample_rate != self.TARGET_SAMPLE_RATE:
            signal = torchaudio.transforms.signalResample(sample_rate,self.TARGET_SAMPLE_RATE)(signal)
        return signal

    @classmethod
    def samplingTransformClass(cls,signal,sample_rate):
        #Usamos sample_rate 16000 por defecto
        if sample_rate != cls.TARGET_SAMPLE_RATE:
            signal = torchaudio.transforms.signalResample(sample_rate,cls.TARGET_SAMPLE_RATE)(signal)
        return signal
            
    @classmethod        
    def monoTransformClass(cls, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0,keepdim=True)
        return signal

    def monoTransform(self,signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0,keepdim=True)
        return signal
    
    def setTargetSampleRate(self,sample_rate):
        self.TARGET_SAMPLE_RATE=sample_rate
        return 0

    def setTransformation(self,transformation):
        self.TRANSFORMATION=transformation
        return 0


if __name__ == "__main__":
    # audio samples are loaded as an int16 numpy array
    # rescale intensity range as float [-1, 1]
    toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
    # use instrument_family and instrument_source as classification targets
    dataset = NSynth(
        "../nsynth-test",
        transform=toFloat,
        blacklist_pattern=["string"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    for samples, instrument_family_target, instrument_source_target, targets \
            in loader:
        print(samples.shape, instrument_family_target.shape,
              instrument_source_target.shape)
        print(torch.min(samples), torch.max(samples))
