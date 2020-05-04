import time


class Sleep:
    def __init__(self, wait):
        self.wait = wait

    def __enter__(self):
        self.start = self.__t()
        self.finish = self.start + self.wait

    def __exit__(self, value, traceback):
        while self.__t() < self.finish:
            time.sleep(1. / 1000.)

    def __t(self):
        return int(round(time.time() * 1000))