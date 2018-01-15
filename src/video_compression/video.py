import cv2


class VideoReadError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

class Video(object):
    def __init__(self,video_filename):
        self.video = cv2.VideoCapture(video_filename)
        if not self.video.isOpened():
            raise VideoReadError(f"An unknown error is happened while opening '{video_filename}' with opencv.")
        self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_channels = 3
        self.frames = []
        self.__get_all_frames()

    def __get_all_frames(self):
        while True:
            flag, frame = self.video.read()
            if flag:
                self.frames.append(frame)
            else:
                break

    def get_frames(self, index_range):
        if index_range == "all":
            return self.frames
        else:
            return self.frames[index_range]

    def __del__(self):
        self.video.release()