import cv2
from PIL import ImageTk, Image


class VideoCapture:
    _instance = None
    root = None
    video = None
    video_source = None
    width = 0
    height = 0
    CV_SYSTEM_CACHE_CNT = 3

    @staticmethod
    def instance():
        if VideoCapture._instance is None:
            VideoCapture._instance = VideoCapture()
        return VideoCapture._instance

    def set_root(self, window):
        self.root = window

    def set_source(self, source):
        self.video_source = source

    def __init__(self):
        if VideoCapture._instance is None:
            VideoCapture._instance = self

    def get_value(self):
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def start(self):
        if self.check_active() is True:
            if self.video is None:
                self.video = cv2.VideoCapture(self.video_source)
                self.get_value()
            else:
                if self.video.isOpened():
                    self.video.release()
                    self.video = cv2.VideoCapture(self.video_source)
                    self.get_value()
                else:
                    self.video = cv2.VideoCapture(self.video_source)
                    self.get_value()
        else:
            print("Source hasn't been set")
            return

    def stop(self):
        if self.check_active() is True:
            if self.video is not None:
                if self.video.isOpened():
                    self.video.release()
                    print("RELEASE VIDEO")
            else:
                return
        else:
            print("Source hasn't been set")
            return

    def get_frame(self):
        if self.check_active() is False:
            return False, None
        elif self.video.isOpened():
            for i in range(0, self.CV_SYSTEM_CACHE_CNT):
                self.video.read()
            ret, frame = self.video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = ImageTk.PhotoImage(image=Image.fromarray(frame))
                return ret, frame
            else:
                return ret, None
        else:
            return False, None

    def check_active(self):
        if self.video_source is None:
            return False
        else:
            return True

    def __del__(self):
        if self.video is None:
            return
        elif self.video.isOpened():
            self.video.release()
            self.video = None
