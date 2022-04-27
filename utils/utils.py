import os
import torch
import torchaudio
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from .nsynth import NSynth, SignalTransformation
import sys
import librosa.display
import json
import matplotlib.pyplot as plt

def organize_images(path):
    ficheros = os.listdir(path)
    pathInicial = path[:]

    for i in ficheros:
        if "png" in i:
            classification = i.index("_")
            directorio = i[:classification]

            nuevoPath = pathInicial + directorio + "/"

            try:
                os.rename(path + i, nuevoPath + i)
            except:
                if not os.path.exists(nuevoPath):
                    os.makedirs(nuevoPath)
                    os.rename(path + i, nuevoPath + i)


#file_route: ruta al archivo examples.json en el que estan recogidos los datos de las muestras
def generate_spectrogram(file_route):
    with open(file_route, "r") as f:
        datos = json.loads(f.read())

    for dato in datos:
        waveform, sample_rate = torchaudio.load("./nsynth-test/audio/" + dato + ".wav")
        waveform = waveform.to(device)
        waveform = SignalTransformation.generarSpectrogramaFromSignal(waveform)
        shape = waveform.shape
        waveform = torchaudio.transforms.AmplitudeToDB()(waveform)
        waveform = waveform.cpu().data.numpy()
        librosa.display.specshow(waveform[0], cmap='magma')
        plt.savefig("./nsynth-test/MelSpectrograms/" + dato + ".png", bbox_inches='tight')
        plt.close()

#audio_path: ruta al archivo de audio en .wav
#save_path: directorio en el que se guardan las muestras
#window_duration: salto de tiempo en segundos
def separar(audio_path, save_path, window_duration):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to('cuda:0')
    dimension = waveform.size(dim=1)

    anterior = 1
    i=0
    k=window_duration * sample_rate
    while (anterior+sample_rate*4) <= dimension:
            audio = torch.narrow(waveform, dim=1, start = anterior, length = sample_rate*4)
            anterior = anterior + k
            #print(prueba.size(dim=1))

            audio = SignalTransformation.generarSpectrogramaFromSignal(audio)
            shape = audio.shape
            audio = torchaudio.transforms.AmplitudeToDB()(audio)
            audio=audio.cpu().data.numpy()
            librosa.display.specshow(audio[0],cmap='magma')
            nombre = save_path+str(i)+".png"
            plt.savefig(nombre, bbox_inches='tight')
            plt.close()
            i = i+1




