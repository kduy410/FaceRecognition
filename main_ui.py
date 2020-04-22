import os
import subprocess
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Style
import sys


class GUI(Frame):
    DATA_PART = ""
    WEIGHTS = ""

    def __init__(self):
        super().__init__()
        self.log = Logger()
        self.frm_main = Frame(master=self, bg="red", width=600, height=400)
        self.frm_right = Frame(master=self, bg="green", width=120, height=400, relief=RAISED, borderwidth=1)
        self.frm_bottom = Frame(master=self, bg="blue", width=600, height=40)
        self.frm_console = Frame(master=self, bg="white", width=600, height=80)

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
                               fg="black")

        self.btn_capture = Button(master=self.frm_bottom, text="Capture",
                                  bg="white",
                                  fg="black")
        self.btn_right = Button(master=self.frm_bottom, text=">>>",
                                bg="white",
                                fg="black")
        self.scroll_bar = Scrollbar(master=self.frm_console, orient=VERTICAL)
        self.eula = Text(master=self.frm_console, font=('Arial', 10), wrap=None,
                         yscrollcommand=self.scroll_bar.set, undo=True)
        self.scroll_bar.config(command=self.eula.yview)

        self.master.title("FACE DETECTION SOFTWARE")
        self.style = Style()
        self.style.theme_use("default")
        self.style.configure(style="TButton", font=('FreeSens', 5))

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
                print("You choose %s" % file_path)
                if os.path.isfile(file_path):
                    self.WEIGHTS = file_path
                else:
                    print("This is not a file")
                    return
        except AttributeError as ae:
            print(str(ae))

        self.print_to_GUI()


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
    # root.geometry("720x520+300+0")
    application = GUI()
    root.mainloop()


if __name__ == '__main__':
    main()
