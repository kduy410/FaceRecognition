import os
import subprocess
from tkinter import *
from tkinter import filedialog
from tkinter.font import Font
from tkinter.ttk import Style
import sys
import cv2
from PIL import ImageTk, Image


class GUI(Frame):
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.wmv', '.flv')
    IMAGES = []
    image = None
    index = 0
    DATA_PART = ""
    WEIGHTS = ""
    image_size = 221, 221

    def __init__(self):
        super().__init__()
        self.log = Logger()
        self.frm_main = Frame(master=self, bg="red", width=720, height=480)
        self.frm_right = Frame(master=self, bg="green", width=120, height=480, relief=RAISED, borderwidth=1)
        self.frm_bottom = Frame(master=self, bg="blue", width=720, height=40)
        self.frm_console = Frame(master=self, bg="white", width=720, height=80)

        self.btn_file = Button(master=self.frm_right, text="File",
                               bg="white",
                               fg="black")
        self.btn_load_data = Button(master=self.frm_right, text="Load data",
                                    bg="white",
                                    fg="black",
                                    command=self.open_file)
        self.btn_train_data = Button(master=self.frm_right, text="Train data",
                                     bg="white",
                                     fg="black")
        self.btn_video = Button(master=self.frm_right, text="Video",
                                bg="white",
                                fg="black")
        self.btn_camera = Button(master=self.frm_right, text="Camera",
                                 bg="white",
                                 fg="black")

        self.btn_folder = Button(master=self.frm_right, text="Folder",
                                 bg="white",
                                 fg="black")
        self.btn_left = Button(master=self.frm_bottom, text="<<<",
                               bg="white",
                               fg="black",
                               command=self.left)

        self.btn_capture = Button(master=self.frm_bottom, text="Capture",
                                  bg="white",
                                  fg="black")
        self.btn_right = Button(master=self.frm_bottom, text=">>>",
                                bg="white",
                                fg="black",
                                command=self.right)
        self.scroll_bar = Scrollbar(master=self.frm_console, orient=VERTICAL)
        self.eula = Text(master=self.frm_console, font=('Arial', 10), wrap=None,
                         yscrollcommand=self.scroll_bar.set, undo=True)
        self.scroll_bar.config(command=self.eula.yview)

        self.master.title("FACE DETECTION SOFTWARE")
        self.style = Style()
        self.style.theme_use("default")
        self.style.configure(style="TButton", font=('FreeSens', 5))
        self.lbl_image = Label(master=self.frm_main, width=720, height=480)
        self.init_ui()

    def init_ui(self):
        self.frm_right.columnconfigure([0], weight=1)
        self.frm_right.rowconfigure([0, 1, 2, 3, 4, 5], weight=0, pad=10)

        self.btn_train_data.grid(row=0, column=0, padx=3, sticky='ew')
        self.btn_load_data.grid(row=1, column=0, padx=3, sticky='ew')
        self.btn_file.grid(row=2, column=0, padx=3, sticky='ew')
        self.btn_folder.grid(row=3, column=0, padx=3, sticky='ew')
        self.btn_video.grid(row=4, column=0, padx=3, sticky='ew')
        self.btn_camera.grid(row=5, column=0, padx=3, sticky='ew')

        self.frm_bottom.columnconfigure([0, 1, 2], weight=1, pad=10)
        self.frm_bottom.rowconfigure([0], weight=1, pad=20)

        self.frm_console.columnconfigure([0], weight=1)
        self.frm_console.columnconfigure([1], weight=0)
        self.frm_console.grid_propagate(False)

        self.eula.grid(row=0, column=0, sticky="nsew")
        self.scroll_bar.grid(row=0, column=1, sticky="ns")

        self.btn_left.grid(row=0, column=0, padx=10, sticky='ew')
        self.btn_capture.grid(row=0, column=1, padx=5, sticky='ew')
        self.btn_right.grid(row=0, column=2, padx=10, sticky='ew')

        self.lbl_image.pack(fill=BOTH, side=TOP, expand=True)
        self.frm_console.pack(fill=BOTH, side=BOTTOM, expand=True)
        self.frm_right.pack(fill=BOTH, side=RIGHT, expand=True)
        self.frm_bottom.pack(fill=BOTH, side=BOTTOM, expand=True)
        self.frm_main.pack(fill=BOTH, side=LEFT, expand=True)
        self.pack(fill=BOTH, expand=True)

    def print_to_GUI(self):
        self.eula.insert(END, str(self.log.messages) + "\n")
        self.log.messages.clear()
        self.log.stop()

    def open_file(self):
        self.log.start()
        file_path = filedialog.askopenfilename(parent=self.master, title="Please select a file")
        try:
            if len(file_path) > 0:
                print(file_path)
                if os.path.isfile(file_path):
                    if file_path.lower().endswith(self.IMAGE_EXTENSIONS):
                        image = cv2.imread(file_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (720, 480), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                        self.append(image)
                        self.image = ImageTk.PhotoImage(image=Image.fromarray(image))
                        self.lbl_image.config(anchor='nw', image=self.image)

                    elif file_path.lower().endswith(self.VIDEO_EXTENSIONS):
                        # do something
                        return
                    else:
                        print("Unknown file type, please choose again")
                        self.print_to_GUI()
                        return
        except AttributeError as ae:
            print(str(ae))
        self.print_to_GUI()

    def append(self, image):
        self.IMAGES.append(image)
        self.index = len(self.IMAGES) - 1

    def left(self):
        print("You click left")
        for i, _ in enumerate(self.IMAGES):
            if i == self.index:
                if i == 0:
                    print('This is the first image')
                    return
                else:
                    self.index = self.index - 1
                    self.image = ImageTk.PhotoImage(image=Image.fromarray(self.IMAGES[self.index]))
                    self.lbl_image.config(anchor='nw', image=self.image)
                    break

    def right(self):
        print('You click right')
        for i, _ in enumerate(self.IMAGES):
            if i == self.index:
                if self.index + 1 == len(self.IMAGES):
                    print('This is the last image')
                    return
                else:
                    self.index = self.index + 1
                    self.image = ImageTk.PhotoImage(image=Image.fromarray(self.IMAGES[self.index]))
                    self.lbl_image.config(anchor='nw', image=self.image)
                    break


class Logger:
    stdout = sys.stdout
    messages = []

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.stdout

    def write(self, text):
        self.messages.append(text)


def main():
    root = Tk()
    root.geometry("840x600+300+0")
    application = GUI()
    root.mainloop()


if __name__ == '__main__':
    main()
