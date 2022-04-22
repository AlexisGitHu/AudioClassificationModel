from tkinter import *
import easygui
import os

from PIL import ImageTk, Image

classes = []

running = True


def quit_me():
    global running
    running = False
    pressed.set(pressed.get())
    titleWindow.destroy()


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

background_img = ImageTk.PhotoImage(Image.open("Images/Background.jpg").resize((750, 500), Image.Resampling.LANCZOS))
background_label = Label(titleWindow, image=background_img)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# main_img = Image.open("App Images/Main.png")
# mwidth, mheight = main_img.size
# main_img = ImageTk.PhotoImage(main_img.resize((mwidth // 2, mheight // 2), Image.ANTIALIAS))
# main_label = Label(titleWindow, image=main_img)
# main_label.pack(pady=12, anchor="center")

pressed = IntVar()
btn = Button(titleWindow, text="Seleccionar\n Pista de Audio \n/ Espectrograma", font=("Arial", 23), width=15, height=3,
             bg="#FFFFFF", activebackground="#E2E2E2", relief="solid", border=2, command=lambda: pressed.set(1))

btn.pack(side=BOTTOM, pady=50)

btn.wait_variable(pressed)

while running:
    desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop/')

    sample_path = str(
        easygui.fileopenbox(title="Seleccionar Pista de Audio / Espectrograma",
                            default=desktop_path,
                            filetypes=[["*.png", "*.jpg", "Image Files"], ["*.wav", "Audio Files"]],
                            multiple=False))

    # main_label.destroy()
    btn.destroy()

    extension = os.path.splitext(sample_path)[1]

    if extension in [".wav"]:
        print("Pista de Audio")

    elif extension in [".jpg", ".png"]:
        print("Espectrograma")

    else:
        print("Archivo no permitido")

    # food_img = Image.open(food_img_path)
    # food_img = ImageTk.PhotoImage(food_img.resize((250, 250), Image.ANTIALIAS))
    # food_label = Label(titleWindow, image=food_img, borderwidth=2, relief="solid")
    # food_label.pack(pady=14, anchor="center")
    # text = "classify(food_img_path)"

    pressed = IntVar()
    btn2 = Button(titleWindow, text="Seleccionar Nueva\nPista", font=("Arial", 16), width=16, height=2,
                  bg="#FFFFFF", activebackground="#E2E2E2", relief="solid", border=2, command=lambda: pressed.set(1))

    btn2.pack(side=BOTTOM, pady=50)

    btn2.wait_variable(pressed)

    if running:
        # food_label.destroy()
        # text.destroy()
        btn2.destroy()

titleWindow.mainloop()
