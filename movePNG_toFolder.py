import os

path = 'C:/Users/david/Desktop/U-TAD/3er AÑO/Proyecto/spectrograms/'

ficheros = os.listdir(path)

pathInicial = path[:]

for i in ficheros:
    if "png" in i:
        classification = i.index("_")
        
        directorio = i[:classification]

        nuevoPath = pathInicial+directorio+"/"

        os.rename(path+i, nuevoPath+i)

