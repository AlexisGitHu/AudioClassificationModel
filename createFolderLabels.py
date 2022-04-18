import os

path = 'C:/Users/david/Desktop/U-TAD/3er AÑO/Proyecto/spectrograms/'

ficheros = os.listdir(path)

pathInicial = path[:]

for i in ficheros:
    classification = i.index("_")
    
    directorio = i[:classification]

    nuevoPath = pathInicial+directorio

    isExist = os.path.exists(nuevoPath)


    if not isExist:
        os.makedirs(nuevoPath)
        print("The new directory is created!")