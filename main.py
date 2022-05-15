import shutil
from tkinter import *
import easygui
import os
from utils import utils
from CNN_musical import CNN_musical as CNN
from pathlib import Path
from PIL import ImageTk, Image
from playsound import playsound

classes = []

running = True


def quit_me():
    global running
    running = False
    pressed.set(pressed.get())
    titleWindow.destroy()


model = CNN.cargarModelo("trainedModelNoHistory.pth", "cuda:0")

titleWindow = Tk()
titleWindow.protocol("WM_DELETE_WINDOW", quit_me)
titleWindow.resizable(False, False)
titleWindow.title("Audio Classifier")

titleWindow.geometry("750x500")
Tk_Width = 750
Tk_Height = 600
x_Left = int(titleWindow.winfo_screenwidth() / 2 - Tk_Width / 2)
y_Top = int(titleWindow.winfo_screenheight() / 2 - Tk_Height / 2)
titleWindow.geometry("+{}+{}".format(x_Left, y_Top))

background_img = ImageTk.PhotoImage(Image.open("AudioClassificationApp/Images/Background.jpg").resize((750, 500)))
background_label = Label(titleWindow, image=background_img)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

pressed = IntVar()
btn = Button(titleWindow, text="Seleccionar\n Pista de Audio \n/ Espectrograma", font=("Arial", 23), width=15, height=3,
             bg="#FFFFFF", activebackground="#E2E2E2", relief="solid", border=2, command=lambda: pressed.set(1))

btn.pack(side=BOTTOM, pady=50)

btn.wait_variable(pressed)

while running:
    sample_path = str(
        easygui.fileopenbox(title="Seleccionar Pista de Audio / Espectrograma",
                            filetypes=[["*.png", "*.jpg", "Image Files"], ["*.wav", "Audio Files"]],
                            multiple=False))

    btn.destroy()

    background_img = ImageTk.PhotoImage(
        Image.open("AudioClassificationApp/Images/Background.jpg").resize((750, 500)))
    background_label = Label(titleWindow, image=background_img)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)


    def play():
        playsound(sample_path)


    extension = os.path.splitext(sample_path)[1]
    file_name = Path(sample_path).name

    if extension in [".wav"]:
        save_path = "AudioClassificationApp/AudiosSeparados/" + file_name + "/Espectrogramas/"

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        os.makedirs(save_path)

        utils.separar(sample_path, save_path, 4)

        play_button = Button(titleWindow, text="Reproducir Pista de Audio", font=("Arial", 20),
                             bg="#FFFFFF", activebackground="#E2E2E2", relief="solid", border=2, command=play)
        play_button.pack(pady=16)

        spectrogram_image = ImageTk.PhotoImage(Image.open(save_path + "/0.png").resize((260, 197)))
        spectrogram_label = Label(titleWindow, image=spectrogram_image, borderwidth=2, relief="solid")
        spectrogram_label.pack(pady=8)

        try:
            resultado = CNN.most_common(CNN.predict(model, "AudioClassificationApp/AudiosSeparados/" + file_name + "/", "cpu"))
        except:
            print("Vaya")

        result_label = Label(titleWindow, text="Resultado: " + resultado, font=("Arial", 25), border=2, relief="solid")
        result_label.pack(pady=16, ipadx=6, ipady=4)

    elif extension in [".jpg", ".png"]:
        spectrogram_image = ImageTk.PhotoImage(Image.open(sample_path).resize((260, 197)))
        spectrogram_label = Label(titleWindow, image=spectrogram_image, borderwidth=2, relief="solid")
        spectrogram_label.pack(pady=(95, 8))

        result_label = Label(titleWindow, text="Resultado: Guitarra", font=("Arial", 25), border=2, relief="solid")
        result_label.pack(pady=16, ipadx=6, ipady=4)

    else:
        quit_me()

    pressed = IntVar()
    btn2 = Button(titleWindow, text="Seleccionar Nueva\nPista", font=("Arial", 16), width=16, height=2,
                  bg="#FFFFFF", activebackground="#E2E2E2", relief="solid", border=2, command=lambda: pressed.set(1))

    btn2.pack(side=BOTTOM, pady=20)

    btn2.wait_variable(pressed)

    if running:
        if extension in [".wav"]:
            play_button.destroy()

        result_label.destroy()
        spectrogram_label.destroy()
        btn2.destroy()

titleWindow.mainloop()
