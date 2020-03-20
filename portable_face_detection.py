import sys
import time
import tkinter
from tkinter import filedialog

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
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_DUPLEX
MODEL = "hog"  # HOG if use CPU - ONLY ELSE use CNN WITH CUDA
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
RED = [0, 0, 255]

faces = {}  # dict
faces_encoded = None
known_face_names = None

# Duyet windows tim duong dan den thu muc
root = tkinter.Tk()
root.withdraw()


def browse():
    current_dir = os.getcwd()
    file_path = filedialog.askdirectory(parent=root, title='Please select a directory')
    try:
        if len(file_path) > 0:
            print("You choose %s" % file_path)
            return file_path
    except AttributeError as ae:
        print(str(ae))


def browse_windows_file():
    file_path = filedialog.askopenfilename()
    print(file_path)
    try:
        kind = filetype.guess(file_path)
        if filetype.is_extension_supported(kind.extension) is False:
            print("This extension isn't supported")
            return None
        else:
            return file_path
    except AttributeError as ae:
        print(str(ae))


def browse_windows_folder():
    path = filedialog.askdirectory()
    print(path)
    try:
        if os.path.isdir(path) is False:
            print("This extension isn't supported")
            return None
        else:
            return path
    except AttributeError as ae:
        print(str(ae))


def pre_process():
    global faces, faces_encoded, known_face_names
    faces = get_encoded_faces()  # dict
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())


def get_encoded_faces():
    """
        looks through the folder and encodes all
        the faces

        :return: dict of (name, image encoded)
    """
    global KNOWN_FACES_DIR
    # KNOWN_FACES_DIR = browse()
    KNOWN_FACES_DIR = "D:/Data/imagesKPOP"
    if not os.path.isdir(KNOWN_FACES_DIR):
        return
    else:
        encoded = {}
        for name in tqdm(os.listdir(KNOWN_FACES_DIR)):
            for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
                    # print(f"{KNOWN_FACES_DIR}/{name}/{filename}")
                    encoding = face_recognition.face_encodings(image)[0]  # FIRST FACE IT FIND
                    # encoded[filename.split(".")[0]] = encoding
                    encoded[name] = encoding

                    # known_faces.append(encoding)
                    # known_names.append(name)

        return encoded


def unknown_image_encoded(name, image):
    """
    encode a face given the file name
    """
    face = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{image}")
    encoding = face_recognition.face_encodings(face)[0]
    return encoding


print("PROCESSING UNKNOWN FACES")


def classify_face(img):
    # print(original)
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    if it is camera parameter = 1
             image  parameter = 0
    """
    # faces = get_encoded_faces()  # dict
    # faces_encoded = list(faces.values())
    # known_face_names = list(faces.keys())
    global faces_encoded, known_face_names
    # img = face_recognition.load_image_file(im)
    # img = cv2.imread(img, cv2.IMREAD_COLOR)  # it will return error because imread need a path to read an image
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # DINH DANG BGR CUA OPENCV
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # img = img[:, :, ::-1]

    face_locations = face_recognition.face_locations(img, model=MODEL)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
    face_names = []
    packet = {}
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        start = time.process_time()

        print(f"\nStart: {start:.10f}")
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        end = time.process_time()
        print(f"\nEnd: {end:.10f}")
        packet = zip(face_locations, face_names, str(end - start))
        # print(packet)

    return packet


def display(name, image):
    while True:
        image = imutils.resize(image, width=720, height=1280)
        cv2.imshow(f'{name}', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return


def analysing_image():
    global UNKNOWN_FACES_DIR
    UNKNOWN_FACES_DIR = browse()
    for filename in tqdm(os.listdir(f"{UNKNOWN_FACES_DIR}")):
        image = cv2.imread(f"{UNKNOWN_FACES_DIR}/{filename}", cv2.IMREAD_COLOR)
        packet = classify_face(image)
        # print(image)
        image = draw_box(image, packet)
        # print(image)
        display(f"{UNKNOWN_FACES_DIR}/{filename}", image)


def draw_box(image, packet):
    """
        image: to draw a box
        :parameter packet : a zip packet of face_location and face_name
        :parameter image
        :return an image
    """
    if type(packet) is not zip:  # check the packet if it is tuple else return original image
        print("Can not find face!")
        return image
    for (top, right, bottom, left), name, time in packet:
        # Draw a box around the face
        if name is "Unknown":
            # Draw a box around the face
            cv2.rectangle(image, (left - 20, top - 20), (right + 20, bottom + 20), RED, FRAME_THICKNESS)
            # Draw a label with a name below the face
            cv2.rectangle(image, (left - 20, bottom - 15), (right + 20, bottom + 20), RED, cv2.FILLED)
            # Put text
            cv2.putText(image, name, (left - 20, bottom + 20), FONT, 1.0, (255, 255, 255), 2)
            cv2.putText(image, time, (left - 20, bottom + 40), FONT, 1.0, (255, 255, 255), 2)
        else:
            cv2.rectangle(image, (left - 20, top - 20), (right + 20, bottom + 20), GREEN, FRAME_THICKNESS)
            cv2.putText(image, name, (left - 20, bottom + 20), FONT, 1.0, RED, 2)
            cv2.putText(image, time, (left - 20, bottom + 40), FONT, 1.0, RED, 2)
    return image


def camera():
    cap = cv2.VideoCapture(0)
    print(cap.get(3), "x", cap.get(4))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            packet = classify_face(frame)
            print("Frame: ", cap.get(cv2.CAP_PROP_FPS))
            frame = draw_box(image=frame, packet=packet)
            cv2.imshow("Camera", frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    pre_process()
    analysing_image()
    # camera()


if __name__ == "__main__":
    main()
