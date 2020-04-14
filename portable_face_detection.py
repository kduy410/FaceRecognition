import pickle
import sys
import time
import tkinter
from tkinter import filedialog

import PIL
from PIL import ImageTk, Image
import dlib
import filetype
import face_recognition
import os
import cv2
import imutils
import numpy as np
from tqdm import tqdm

KNOWN_FACES_DIR = f"D:/Data/imagesKPOP"
UNKNOWN_FACES_DIR = f"D:/Data/test"
TOLERANCE = 0.6  # The higher -> more matches, the lower < fault negative
FRAME_THICKNESS = 1
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_DUPLEX
MODEL = "hog"  # HOG if use CPU - ONLY ELSE use CNN WITH CUDA
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
RED = [0, 0, 255]
IMG_SIZE = 70

known_faces = []
known_names = []
next_id = 0


def open_file():
    global UNKNOWN_FACES_DIR
    file_path = filedialog.askopenfilename(parent=root, title="Please select a file or folder")
    try:
        if len(file_path) > 0:
            print("You choose %s" % file_path)
            if os.path.isfile(file_path):
                UNKNOWN_FACES_DIR = file_path
                analysing_image()
            else:
                print("This is not a file")
                return
    except AttributeError as ae:
        print(str(ae))


def open_folder():
    global UNKNOWN_FACES_DIR
    file_path = filedialog.askdirectory(parent=root, title="Please select a file or folder")
    try:
        if len(file_path) > 0:
            print("You choose %s" % file_path)
            if os.path.isdir(file_path):
                UNKNOWN_FACES_DIR = file_path
                analysing_image()
            else:
                print("This is not a file")
                return
    except AttributeError as ae:
        print(str(ae))


def browse():
    root.withdraw()
    file_path = filedialog.askopenfilename(parent=root, title="Please select a file or folder")
    try:
        if len(file_path) > 0:
            print("You choose %s" % file_path)
            if os.path.isfile(file_path):
                return file_path
            else:
                return os.path.dirname(file_path)
    except AttributeError as ae:
        print(str(ae))


def camera():
    cap = cv2.VideoCapture(0)
    print(cap.get(3), "x", cap.get(4))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = classify_face(frame)
            print("Frame: ", cap.get(cv2.CAP_PROP_FPS))
            # frame = draw_box(image=frame, packet=packet)
            cv2.imshow("Camera", frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def video():
    global UNKNOWN_FACES_DIR

    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    file_path = filedialog.askopenfilename(parent=root, title="Please select a file or folder")
    try:
        if len(file_path) > 0:
            print("You choose %s" % file_path)
            if os.path.isfile(file_path) and file_path.endswith(".mp4"):
                cap = cv2.VideoCapture(file_path)
                print(cap.get(3), "x", cap.get(4))
                count = 0
                while True:
                    ret, frame = cap.read()
                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    if cv2.waitKey(10) & 0xFF == 27:
                        break
                    if ret:
                        # frame = face_recognition.load_image_file(frame, mode="L")
                        frame = classify_face(frame)
                        cv2.imshow(file_path, frame)
                cap.release()
                cv2.destroyAllWindows()
            else:
                print("This is not a file")
                return
    except AttributeError as ae:
        print(str(ae))


def load_data():
    global KNOWN_FACES_DIR
    folder_path = filedialog.askdirectory(parent=root, title="Please select a folder")

    try:
        if len(folder_path) > 0:
            print("You choose %s" % folder_path)
            KNOWN_FACES_DIR = folder_path
            load_faces()
        else:
            raise print("Folder empty!")
    except AttributeError as ae:
        print(str(ae))


root = tkinter.Tk()
root.title("FACE DETECTION")
load_data = tkinter.Button(root, text="LOAD DATA", command=load_data).pack()
open_file = tkinter.Button(root, text="Open File", command=open_file).pack()
open_folder = tkinter.Button(root, text="Open Folder", command=open_folder).pack()
video = tkinter.Button(root, text="Open Video", command=video).pack()
camera = tkinter.Button(root, text="Open Camera", command=camera).pack()


def load_faces():
    """
        looks through the folder and encodes all
        the faces
        :return: dict of (name, image encoded)
    """
    global next_id

    if not os.path.isdir(KNOWN_FACES_DIR):
        print("This is not a folder")
        return
    else:
        print("LOADING KNOWN FACES")
        for name in tqdm(os.listdir(KNOWN_FACES_DIR)):
            for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
                    encoding = face_recognition.face_encodings(image)[0]  # FIRST FACE IT FIND
                    known_faces.append(encoding)
                    known_names.append(name)
                elif filename.endswith(".pkl"):
                    encoding = pickle.load(open(f"{KNOWN_FACES_DIR}/{name}/{filename}", "rb"))  # FIRST FACE IT FIND
                    known_faces.append(encoding)
                    known_names.append(int(name))
    if len(known_names) > 0:
        next_id = len(known_names) + 1
    else:
        next_id = 0


def classify_face(image):
    # print(original)
    """
    will find all of the faces in a given image and label
    them if it knows what they are
    """
    print("\nPROCESSING UNKNOWN FACES")

    global next_id
    # image = face_recognition.load_image_file(image, mode="L")
    # image = np.array(image)
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        # See if the face is a match for the known face(s)
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)

        if results[best_match_index]:
            match = known_names[best_match_index]
            print(f"\nMatch found :{match}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            # Draw a box around the face
            cv2.rectangle(image, top_left, bottom_right, GREEN, FRAME_THICKNESS)

            # Draw a label with a name below the face
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 20)
            cv2.rectangle(image, top_left, bottom_right, GREEN, cv2.FILLED)
            # Put text
            cv2.putText(image, str(match), (face_location[3] + 20, face_location[2] + 20), FONT, 1.0, (255, 255, 255),
                        2)
        else:
            match = str(next_id)
            next_id += 1
            known_faces.append(face_encoding)
            known_names.append(match)
            os.makedirs(f"{KNOWN_FACES_DIR}/{match}")
            pickle.dump(face_encoding, open(f"{KNOWN_FACES_DIR}/{match}/{match}-{int(time.time())}.pkl", "wb"))

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            # Draw a box around the face
            cv2.rectangle(image, top_left, bottom_right, RED, FRAME_THICKNESS)

            # Draw a label with a name below the face
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 20)
            cv2.rectangle(image, top_left, bottom_right, RED, cv2.FILLED)
            # Put text
            cv2.putText(image, match, (face_location[3] + 20, face_location[2] + 20), FONT, 1.0, (255, 255, 255), 2)

    return image


def display(name, image):
    while True:
        image = imutils.resize(image, width=720, height=1280)
        cv2.imshow(f'{name}', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return


def analysing_image():
    if os.path.isfile(UNKNOWN_FACES_DIR):
        image = cv2.imread(f"{UNKNOWN_FACES_DIR}", cv2.IMREAD_COLOR)
        image = classify_face(image)
        display(f"{UNKNOWN_FACES_DIR}", image)
    else:
        for filename in tqdm(os.listdir(f"{UNKNOWN_FACES_DIR}")):
            image = cv2.imread(f"{UNKNOWN_FACES_DIR}/{filename}", cv2.IMREAD_COLOR)
            image = classify_face(image)
            display(f"{UNKNOWN_FACES_DIR}/{filename}", image)


# def draw_box(image, packet):
#     """
#         image: to draw a box
#         :parameter packet : a zip packet of face_location and face_name
#         :parameter image
#         :return an image
#     """
#     if type(packet) is not zip:  # check the packet if it is tuple else return original image
#         print("Can not find face!")
#         return image
#     for (top, right, bottom, left), name, time in packet:
#         # Draw a box around the face
#         if name is "Unknown":
#             # Draw a box around the face
#             cv2.rectangle(image, (left - 20, top - 20), (right + 20, bottom + 20), RED, FRAME_THICKNESS)
#             # Draw a label with a name below the face
#             cv2.rectangle(image, (left - 20, bottom - 15), (right + 20, bottom + 20), RED, cv2.FILLED)
#             # Put text
#             cv2.putText(image, name, (left - 20, bottom + 20), FONT, 1.0, (255, 255, 255), 2)
#         else:
#             cv2.rectangle(image, (left - 20, top - 20), (right + 20, bottom + 20), GREEN, FRAME_THICKNESS)
#             cv2.putText(image, name, (left - 20, bottom + 20), FONT, 1.0, RED, 2)
#     return image


def main():
    # load_faces()
    # analysing_image()
    # camera()

    root.mainloop()


if __name__ == "__main__":
    main()
