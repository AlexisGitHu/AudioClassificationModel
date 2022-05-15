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

# Funcion para organizar todos los espectrogramas por carpetas (labels)
def organize_images(path):
    ficheros = os.listdir(path)
    pathInicial = path[:] # Almacenamos el path del que partimos

    # Tendremos que, para cada espectroagrama, crear una carpeta en caso de que no exista ya una carpeta con su label y mover el espectrograma a dicha carpeta
    for i in ficheros:
        if "png" in i:
            classification = i.index("_")
            directorio = i[:classification] # Calculamos el label que tendrá el directorio de la imagen

            # El nuevo path de la imagen será el pathInicial/directorio/
            nuevoPath = pathInicial + directorio + "/"

            # Tratamos de actualizar la imagen al nuevo path, en caso de que el directorio no exista, crear carpeta y moverlo
            # Si la imagen tiene ruta: pathInicial/<imagen> lo intentamos cambiar a: pathIncial/<directorio>/<imagen>
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

        
def ver_onSet(audio):
    y, sr = librosa.load(audio)
    array_tiempos = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    return array_tiempos

#audio_path: ruta al archivo de audio en .wav
#save_path: directorio en el que se guardan las muestras
#window_duration: salto de tiempo en segundos
def separar(audio_path, save_path, window_duration):
    waveform, sample_rate = torchaudio.load(audio_path)
#     waveform = waveform.to('cuda:0')
    dimension = waveform.size(dim=1)
    
    array_tiempos = ver_onSet(audio_path)
#     tensor_tiempos = torch.from_numpy(array_tiempos)
#     tensor_tiempos = torch.tensor(array_tiempos)

#     for anterior in array_tiempos:
#     i = 0
#     for anterior_float in array_tiempos:
# #         anterior_float = anterior_tensor.item()
#         anterior = int(anterior_float)*sample_rate
#         print(anterior)
#         audio = torch.narrow(waveform, dim=1, start = anterior, length = sample_rate*4)
#         audio = SignalTransformation.generarSpectrogramaFromSignal(audio)
#         shape = audio.shape
#         audio = torchaudio.transforms.AmplitudeToDB()(audio)
#         audio=audio.cpu().data.numpy()
#         librosa.display.specshow(audio[0],cmap='magma')
#         nombre = save_path+str(i)+".png"
#         try:
#             plt.savefig(nombre, bbox_inches='tight')
#             plt.close()
#         except:
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#                 plt.savefig(nombre, bbox_inches='tight')
#                 plt.close()
#         i = i+1
#         print(anterior.item())

#     anterior = 1
    anterior = int(array_tiempos[0])*sample_rate
    i=0
#     k=window_duration * sample_rate
    while (anterior+sample_rate*4) <= dimension:
            audio = torch.narrow(waveform, dim=1, start = anterior, length = sample_rate*4)
#             anterior = anterior + k
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
            anterior = int(array_tiempos[i])*sample_rate




