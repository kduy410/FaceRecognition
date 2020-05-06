import os
import sys
import tkinter as tk
import time
from tkinter import filedialog
from tkinter.ttk import Style

import dlib
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tqdm import tqdm

import vgg_model
from ThreadTask import *
from VideoCapture import *
from vgg import display_one


def down_scale(image):
    scale_percent = 80  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return width, height


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class GUI:
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.pkl')
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.wmv', '.flv')
    VIDEO_WIDTH = 0
    VIDEO_HEIGHT = 0
    IMAGES = []
    MESSAGES = []
    THRESH_HOLD = 0.03

    index = 0
    DATA_PART = ""
    WEIGHTS = ""
    image_size = 221, 221
    VIDEO_SOURCE = -1

    FILE_TYPE_ERROR = "Unknown type, please choose again"

    _instance = None
    image = None
    stop_work = False
    label = None
    queue = None
    lock = None
    thread = None
    frame = None
    last_frame = None

    model = None
    MODEL_PATH = "weights/triplet_models_5_221_40.h5"
    x_train = None
    y_train = None
    data_frame = None
    DATA_FRAME_PATH = "weights/dataframe.zip"
    size = (221, 221)
    embs = None
    EMBS_PATH = "weights/embs540-test.npy"
    X_PATH = "weights/x_test_221_shuffle.npy"
    Y_PATH = "weights/y_test_221_shuffle.npy"
    hog = None
    is_saving = False
    SAVE_PATH = "D:/Data/temp/"

    @staticmethod
    def instance(window, window_title):
        if GUI._instance is None:
            GUI._instance = GUI(window, window_title)
        return GUI._instance

    def __init__(self, window, window_title):
        if GUI._instance is None:
            GUI._instance = self

        self.window = window
        self.window.title(window_title)
        self.window.option_add('*tearOff', False)
        self.menu = tk.Menu(master=self.window)
        self.window.config(menu=self.menu)

        file = tk.Menu(self.menu)
        file.add_command(label="Save", command=self.save)
        file.add_command(label="Exit", command=self.quit)
        self.menu.add_cascade(label="File", menu=file)

        self.frm_main = tk.Frame(master=self.window, bg="red", width=720, height=600)
        self.frm_right = tk.Frame(master=self.window, bg="green", width=120, height=600,
                                  relief=tk.RAISED, borderwidth=0)
        self.frm_bottom = tk.Frame(master=self.window, bg="blue", width=720, height=40)
        self.frm_console = tk.Frame(master=self.window, bg="white", width=720, height=80)

        self.btn_file = tk.Button(master=self.frm_right, text="File",
                                  bg="white",
                                  fg="black",
                                  command=self.open_file)
        self.btn_load_data = tk.Button(master=self.frm_right, text="Load data",
                                       bg="white",
                                       fg="black",
                                       command=self.load_data)
        # self.btn_train_data = tk.Button(master=self.frm_right, text="Train data",
        #                                 bg="white",
        #                                 fg="black")
        self.btn_video = tk.Button(master=self.frm_right, text="Video",
                                   bg="white",
                                   fg="black",
                                   command=self.open_video)
        self.btn_camera = tk.Button(master=self.frm_right, text="Camera",
                                    bg="white",
                                    fg="black",
                                    command=self.open_camera)

        self.btn_folder = tk.Button(master=self.frm_right, text="Folder",
                                    bg="white",
                                    fg="black",
                                    command=self.open_folder)
        self.btn_left = tk.Button(master=self.frm_bottom, text="<<<",
                                  bg="white",
                                  fg="black",
                                  command=self.left)

        self.btn_capture = tk.Button(master=self.frm_bottom, text="Capture",
                                     bg="white",
                                     fg="black",
                                     command=self.capture)
        self.btn_right = tk.Button(master=self.frm_bottom, text=">>>",
                                   bg="white",
                                   fg="black",
                                   command=self.right)

        self.scroll_bar = tk.Scrollbar(master=self.frm_console, orient=tk.VERTICAL)
        self.eula = tk.Text(master=self.frm_console, font=("consolas", 12), wrap='word',
                            undo=True, borderwidth=0, cursor="")
        # self.eula = tk.Listbox(master=self.frm_console, font=("consolas", 12), wrap=None
        #                        , borderwidth=0, cursor="")

        self.scroll_bar.config(command=self.eula.yview)
        self.eula.configure(yscrollcommand=self.scroll_bar.set)

        self.label = tk.Label(master=self.frm_main, width=720, height=600)

        self.log = Logger()
        self.load_data()

        self.init_ui()

        self.window.mainloop()

    def init_ui(self):

        self.frm_right.columnconfigure([0], weight=1)
        self.frm_right.rowconfigure([0, 1, 2, 3, 4, 5], weight=0, pad=10)

        # self.btn_train_data.grid(row=0, column=0, padx=3, sticky='ew')
        self.btn_load_data.grid(row=1, column=0, padx=3, sticky='ew')
        self.btn_file.grid(row=2, column=0, padx=3, sticky='ew')
        self.btn_folder.grid(row=3, column=0, padx=3, sticky='ew')
        self.btn_video.grid(row=4, column=0, padx=3, sticky='ew')
        self.btn_camera.grid(row=5, column=0, padx=3, sticky='ew')

        self.frm_bottom.columnconfigure([0, 1, 2], weight=1, pad=10)
        self.frm_bottom.rowconfigure([0], weight=1, pad=20)

        self.frm_console.columnconfigure([0], weight=2)
        self.frm_console.columnconfigure([1], weight=0)
        self.frm_console.grid_propagate(False)

        self.eula.grid(row=0, column=0, sticky="nsew")
        self.scroll_bar.grid(row=0, column=1, sticky="ns")

        # self.eula.pack(side=tk.LEFT, fill=tk.Y,expand=False)
        # self.scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)

        self.btn_left.grid(row=0, column=0, padx=10, sticky='ew')
        self.btn_capture.grid(row=0, column=1, padx=5, sticky='ew')
        self.btn_right.grid(row=0, column=2, padx=10, sticky='ew')

        self.label.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        self.frm_console.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)
        self.frm_right.pack(fill=tk.BOTH, side=tk.RIGHT, expand=False)
        self.frm_bottom.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=False)
        self.frm_main.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

    def get_root(self):
        return self.window

    def get_label(self):
        return self.label

    def capture(self):
        self.stop_worker()
        if not self.IMAGES:
            self.log.start()
            print("EMPTY!")
            self.print_to_gui()
            return
        else:
            image = self.IMAGES[self.index][1]
            faces = self.detect(image)
            images = self.predict_image(image, faces)
            # self.IMAGES[self.index][1] = image
            self.set_label_image(self.index)
            self.display_5(image, images)

    def display_5(self, original, images):
        count = 0
        rows = 2
        cols = 3
        plt.subplot(rows, cols, 1), plt.imshow(original), plt.title(f"Original")
        try:
            images = list(reversed(images))
            for i in range(2, cols * rows + 1):
                if count == 5:
                    break
                print(f"PERSON No.{images[count][0]} | LABEL No.{images[count][3]}")
                name = self.data_frame[(self.data_frame['label'] == images[count][3])].iloc[0, 2]
                plt.subplot(rows, cols, i), plt.imshow(images[count][2]),
                plt.title(f"No.{count}-Name:{name}\nEMB:{images[count][1]}")
                count = count + 1

        except TypeError:
            pass
        except IndexError:
            print(f"PERSON No.{images[0][0]} | LABEL No.{images[0][3]}")
            name = self.data_frame[(self.data_frame['label'] == images[0][3])].iloc[0, 2]
            plt.subplot(2, 3, 2), plt.imshow(images[0][2]),
            plt.title(f"No.{count}-Name:{name}\nEMB:{images[count][1]}")
        finally:
            plt.tight_layout(pad=2.0)
            plt.show()

    def detect(self, image):
        start = time.time()
        faces = self.hog(image, 1)
        end = time.time()
        if len(faces) == 0:
            print("NOT DETECTED")
        print("Hog + SVM Execution time: " + str(end - start))
        return faces

    def face_loc_generator(self, faces):
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            yield (x, y, w, h)

    def draw(self, image, faces):
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            # face = image[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return image

    def print_to_gui(self):
        self.log.stop()
        # self.eula.insert(tk.INSERT, '\n' + str(self.log.messages))
        self.eula.insert(tk.END, '\n' + str(self.log.messages))
        self.eula.see('end')
        self.log.messages.clear()

    def open_file(self):
        file_path = filedialog.askopenfilename(parent=self.window, title="Please select a file")
        self.stop_worker()
        self.log.start()
        try:
            if len(file_path) > 0:
                print(file_path)
                if os.path.isfile(file_path):
                    if file_path.lower().endswith(self.IMAGE_EXTENSIONS):
                        check = self.check_duplicate_image(file_path)
                        if check is None:
                            image = cv2.imread(file_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            self.append([file_path, image])
                            self.set_label_image(self.index)
                            self.print_to_gui()
                        else:
                            print("DUPLICATE")
                            self.index = check
                            self.set_label_image(self.index)
                            self.print_to_gui()
                            return
                    else:
                        print(self.FILE_TYPE_ERROR)

                        return
        except AttributeError as ae:
            print(str(ae))

    def load_data(self):

        try:
            if os.path.exists(self.MODEL_PATH):
                print("Loading model...")
                self.model = load_model(self.MODEL_PATH, compile=False)
                self.model.summary()
                # self.model.compile(optimizer=tf.optimizers.SGD(lr=0.000001,
                #                                                decay=0.001,
                #                                                momentum=0.9,
                #                                                nesterov=True),
                #                    loss=vgg_model._loss_tensor)

                # self.model.compile(optimizer=tf.optimizers.SGD(lr=0.00001,
                #                                                momentum=0.9,
                #                                                nesterov=True),
                #                    loss=vgg_model._loss_tensor)
                # self.model.compile(optimizer=tf.optimizers.SGD(lr=0.0001,
                #                                                momentum=0.9,
                #                                                nesterov=True),
                #                    loss=vgg_model._loss_tensor)
            else:
                print("MODEL NOT FOUND!")
                self.quit()
            print("Loading data frame...")
            if os.path.exists(self.DATA_FRAME_PATH):
                self.data_frame = pd.read_csv(self.DATA_FRAME_PATH)
                print(f"DF LENGTH:\n{len(self.data_frame)}")
            else:
                print("DATA FRAME NOT FOUND!")
                self.quit()

            print("Loading x,y...")
            if os.path.exists(self.X_PATH) & os.path.exists(self.Y_PATH):
                self.x_train = np.load(self.X_PATH)
                self.y_train = np.load(self.Y_PATH)
            else:
                print("X,Y NOT FOUND!")
                self.quit()
            print("Loading emb...")
            if os.path.exists(self.EMBS_PATH):
                self.embs = np.load(self.EMBS_PATH)
            else:
                print("EMB NOT FOUND!")
                self.quit()

            print("Initialize HOG...")
            self.hog = dlib.get_frontal_face_detector()
        except TypeError as te:
            print(str(te))
        except ValueError as ve:
            print(str(ve))
        finally:
            print("DONE")

    def set_label_image(self, index):
        image = self.IMAGES[index][1]
        dim = down_scale(image)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.label.configure(image=self.image)

    def append(self, data):
        self.IMAGES.append(data)
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
                    self.set_label_image(self.index)
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
                    self.set_label_image(self.index)
                    break

    def open_folder(self):
        file_path = filedialog.askdirectory(parent=self.window, title="Please select a folder")
        self.stop_worker()
        self.log.start()
        try:
            if len(file_path) > 0:
                print(f"You choose {file_path}")
                if os.path.isdir(file_path):
                    self.print_to_gui()
                    for file_name in tqdm(os.listdir(f"{file_path}")):
                        full_path = f"{file_path}/{file_name}"
                        check = self.check_duplicate_image(full_path)
                        if check is None:
                            image = cv2.imread(f"{full_path}")
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            self.append([full_path, image])
                        else:
                            continue

                    self.set_label_image(self.index)
                else:
                    print(self.FILE_TYPE_ERROR)
                    return
        except AttributeError as ae:
            print(str(ae))

    def check_duplicate_image(self, path):
        for i, image in enumerate(self.IMAGES):
            if str(path).lower() == str(image[0]).lower():
                return i
        return None

    def open_video(self):
        file_path = filedialog.askopenfilename(parent=self.window, title="Please select a video")
        self.log.start()
        try:
            if len(file_path) > 0:
                print(file_path)
                if os.path.isfile(file_path):
                    if file_path.lower().endswith(self.VIDEO_EXTENSIONS):
                        self.VIDEO_SOURCE = file_path
                        self.print_to_gui()
                        video = VideoCapture.instance()
                        video.set_root(self.window)
                        video.set_source(file_path)

                        self.thread = ThreadTask()
                        self.thread.setDaemon(True)
                        self.thread.start()

                        if self.stop_work is True:
                            self.start_worker()
                            self.worker()
                        else:
                            self.worker()

                    else:
                        print("Unknown video type, please choose again")
                        self.print_to_gui()
                        return
                else:
                    print(self.FILE_TYPE_ERROR)
                    return
        except AttributeError as ae:
            print(str(ae))

    def open_camera(self):
        self.log.start()
        try:
            self.VIDEO_SOURCE = 0
            self.print_to_gui()

            video = VideoCapture.instance()
            video.set_root(self.window)
            video.set_source(self.VIDEO_SOURCE)

            self.thread = ThreadTask()
            self.thread.setDaemon(True)
            self.thread.start()

            if self.stop_work is True:
                self.start_worker()
                self.worker()

            else:
                self.worker()

        except AttributeError:
            self.print_to_gui()

    def worker(self):
        try:
            print(f"\nMAIN")
            self.frame = self.thread.queue.get(False)
            faces = self.detect(self.frame)
            if len(faces) is not 0:
                self.frame = next(self.predict(self.frame, faces))
            self.last_frame = self.frame
        except queue.Empty:
            print("\n\tMAIN QUEUE EMPTY")
            self.set_frame(self.last_frame)
            self.label.update()
        else:
            self.set_frame(self.frame)
            self.label.update()
        finally:
            if self.VIDEO_WIDTH is 0:
                self.get_video_value()
            else:
                pass
            if self.stop_work:
                return
            else:
                self.label.after(LOOP_INTERVAL_TIME, self.worker)

    def set_frame(self, frame):
        if frame is None:
            return
        else:
            self.frame = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.label.configure(image=self.frame)

    def quit(self):
        self.window.quit()
        sys.exit()

    def stop_worker(self):
        if self.stop_work is False:
            self.stop_work = True
            self.window.update()
        else:
            pass

    def start_worker(self):
        if self.stop_work is True:
            self.stop_work = False
            self.window.update()
        else:
            pass

    def get_video_value(self):
        video = VideoCapture.instance()
        if video.width is not 0:
            self.VIDEO_WIDTH = video.width
            self.VIDEO_HEIGHT = video.height
            self.log.start()
            print(f"WIDTH x HEIGHT = {self.VIDEO_WIDTH, self.VIDEO_HEIGHT}")
            self.print_to_gui()
        else:
            pass

    def predict(self, image, faces):
        f = None
        name = None
        emb = None
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            try:
                frame = image[y:y + h, x:x + w]
                frame = cv2.resize(frame, (221, 221))
                if self.is_saving is True:
                    f = frame
                frame = frame / 255.
                frame = np.expand_dims(frame, axis=0)
                emb = self.model.predict([frame, frame, frame])
                minimum = 10
                person = -1
                for i, e in enumerate(self.embs):
                    dist = np.linalg.norm(emb - e)
                    if dist < minimum:
                        minimum = dist
                        person = i
                emb = minimum
                if minimum > self.THRESH_HOLD:
                    name = "Unknown"
                    print(f"\nNAME: {name}")
                    print(f"EMB: {emb}")
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    cv2.putText(image, name, (x, y + h + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    cv2.putText(image, str(round(emb, 6)), (x, y + h + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    name = self.data_frame[(self.data_frame['label'] == self.y_train[person])].iloc[0, 2]
                    print(f"\nPERSON: {person}  LABEL: {self.y_train[person]} NAME: {name}")
                    print(f"EMB: {emb}")
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, name, (x, y + h + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, str(round(emb, 6)), (x, y + h + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except cv2.error:
                pass
            finally:
                if self.is_saving is True:
                    cv2.imwrite(f"{self.SAVE_PATH}/{name}-{emb}.jpg",
                                cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

                yield image

    def predict_image(self, image, faces):
        images = []
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            frame = image[y:y + h, x:x + w]
            frame = cv2.resize(frame, (221, 221))
            frame = frame / 255.
            frame = np.expand_dims(frame, axis=0)

            emb = self.model.predict([frame, frame, frame])
            minimum = 10
            person = -1

            for i, e in enumerate(self.embs):
                dist = np.linalg.norm(emb - e)
                if dist < minimum:
                    minimum = dist
                    person = i
                    images.append([i, minimum, self.x_train[i], self.y_train[i]])
                    print(f"{i} - {minimum}")
            emb = minimum
            if minimum > self.THRESH_HOLD:
                name = "Unknown"
                print(f"\nNAME: {name}")
                print(f"EMB: {emb}")
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.putText(image, name, (x, y + h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.putText(image, str(round(emb, 6)), (x, y + h + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                name = self.data_frame[(self.data_frame['label'] == self.y_train[person])].iloc[0, 2]
                print(f"\nPERSON: {person}  LABEL: {self.y_train[person]} NAME: {name}")
                print(f"EMB: {emb}")
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y + h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, str(round(emb, 6)), (x, y + h + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return images

    def save(self):
        self.log.start()
        if len(self.IMAGES) == 0:
            print("EMPTY!")
            self.print_to_gui()
            return
        else:
            file_path = filedialog.askdirectory(parent=self.window, title="Please select a folder")
            self.stop_worker()
            try:
                if len(file_path) > 0:
                    print(f"You choose {file_path}")
                    if os.path.isdir(file_path):
                        for path, file in tqdm(self.IMAGES):
                            try:
                                name = path.split('/')[-1]
                                cv2.imwrite(f"{file_path}/{name}",
                                            cv2.cvtColor(file, cv2.COLOR_RGB2BGR))
                            except cv2.error:
                                pass
                        print("SAVED!")
                        self.print_to_gui()
                    else:
                        print(self.FILE_TYPE_ERROR)
                        return
            except AttributeError as ae:
                print(str(ae))


class Logger:
    stdout = sys.stdout
    messages = []

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.stdout

    def write(self, text):
        self.messages.append(text)

    def __del__(self):
        print("EXIT")


LOOP_INTERVAL_TIME = 50
CV_SYSTEM_CACHE_CNT = 5
ROOT_REFRESH_RATE = 20


def main():
    root = tk.Tk()
    style = Style()
    style.theme_use("classic")
    root.geometry("840x720+300+0")

    GUI.instance(root, "FACE RECOGNITION")
    root.mainloop()


if __name__ == '__main__':
    main()
