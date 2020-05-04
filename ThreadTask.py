import queue
import threading

from main_ui import VideoCapture


class ThreadTask(threading.Thread):
    queue = queue.Queue(1)
    name = "Video"
    stop_threads = False
    lock = threading.Lock()
    video = VideoCapture.instance()

    def __init__(self):
        super(ThreadTask, self).__init__()

    def run(self):
        try:
            self.video.start()
            self.stop_threads = False
            while True:
                if self.stop_threads:
                    break
                print(f"\nThread {self.name}")
                with self.lock:
                    ret, frame = self.video.get_frame()
                    if ret:
                        self.queue.put(frame)
                    else:
                        continue
        except queue.Full:
            print("\n\tTHREAD QUEUE FULL")
