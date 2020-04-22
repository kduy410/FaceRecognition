import subprocess
from tkinter import *
from tkinter.ttk import Style


class GUI(Frame):

    def __init__(self):
        super().__init__()
        self.master.title("FACE DETECTION SOFTWARE")
        self.style = Style()
        self.style.theme_use("default")
        self.style.configure(style="TButton", font=('FreeSens', 5))
        self.init_ui()

    def init_ui(self):
        self.frm_main = Frame(master=self, bg="red", width=600, height=400)
        frm_right = Frame(master=self, bg="green", width=120, height=400, relief=RAISED, borderwidth=1)
        frm_bottom = Frame(master=self, bg="blue", width=600, height=40)
        frm_console = Frame(master=self, bg="white", width=600, height=80)

        frm_right.columnconfigure([0], weight=1)
        frm_right.rowconfigure([0, 1, 2, 3, 4, 5], weight=0, pad=10)

        btn_train_data = Button(master=frm_right, text="Train data",
                                bg="white",
                                fg="black")

        btn_load_data = Button(master=frm_right, text="Load data",
                               bg="white",
                               fg="black")
        btn_file = Button(master=frm_right, text="File",
                          bg="white",
                          fg="black")
        btn_video = Button(master=frm_right, text="Video",
                           bg="white",
                           fg="black")
        btn_camera = Button(master=frm_right, text="Camera",
                            bg="white",
                            fg="black")

        btn_folder = Button(master=frm_right, text="Folder",
                            bg="white",
                            fg="black")

        btn_train_data.grid(row=0, column=0, padx=3, sticky='ew')
        btn_load_data.grid(row=1, column=0, padx=3, sticky='ew')
        btn_file.grid(row=2, column=0, padx=3, sticky='ew')
        btn_folder.grid(row=3, column=0, padx=3, sticky='ew')
        btn_video.grid(row=4, column=0, padx=3, sticky='ew')
        btn_camera.grid(row=5, column=0, padx=3, sticky='ew')

        frm_bottom.columnconfigure([0, 1, 2], weight=1, pad=10)
        frm_bottom.rowconfigure([0], weight=1, pad=20)

        btn_left = Button(master=frm_bottom, text="<<<",
                          bg="white",
                          fg="black")

        btn_capture = Button(master=frm_bottom, text="Capture",
                             bg="white",
                             fg="black")
        btn_right = Button(master=frm_bottom, text=">>>",
                           bg="white",
                           fg="black")

        frm_console.columnconfigure([0], weight=1)
        frm_console.columnconfigure([1], weight=0)
        frm_console.grid_propagate(False)

        scroll_bar = Scrollbar(master=frm_console, orient=VERTICAL)
        eula = Text(master=frm_console, font=('Arial', 10), wrap=None,
                    yscrollcommand=scroll_bar.set, undo=True)
        eula.insert("1.0", "Hello")
        scroll_bar.config(command=eula.yview)

        eula.grid(row=0, column=0, sticky="nsew")
        scroll_bar.grid(row=0, column=1, sticky="ns")

        btn_left.grid(row=0, column=0, padx=10, sticky='ew')
        btn_capture.grid(row=0, column=1, padx=5, sticky='ew')
        btn_right.grid(row=0, column=2, padx=10, sticky='ew')

        frm_console.pack(fill=BOTH, side=BOTTOM, expand=True)
        frm_right.pack(fill=BOTH, side=RIGHT, expand=True)
        frm_bottom.pack(fill=BOTH, side=BOTTOM, expand=True)
        self.frm_main.pack(fill=BOTH, side=LEFT, expand=True)

        self.pack(fill=BOTH, expand=True)

        proc = subprocess.Popen('a', shell=True, stdout=subprocess.PIPE, )
        output = proc.communicate()[0]
        print(output)




def main():
    root = Tk()
    # root.geometry("720x520+300+0")
    application = GUI()
    root.mainloop()


if __name__ == '__main__':
    main()
