import numpy
import cv2
import random
from math import ceil
from video_compression.sequence import HarmonicSequence


def get_appr_coefficient_V(T, t):
    # using time_averaged vectors V
    return 2 * (T - t + 1) - (T + 1) * (HarmonicSequence.at(T) - HarmonicSequence.at(t - 1))


def get_appr_coefficient_psi(T, t):
    # using \psi directly
    return 2 * t - T - 1


def all_frames():
    return ["all"]


def Mean(n_frames):
    return all_frames()


def Max(n_frames):
    return all_frames()


def Static(n_frames):
    return all_frames()


def SDI(n_frames):
    return all_frames()


def MDI(n_frames, n_segments=3, overlap=0.2):
    n_partial_frames_f = n_frames / (n_segments - n_segments * overlap + overlap)
    n_partial_frames = ceil(n_partial_frames_f)
    segments = []
    start = 0.
    for i in range(n_segments):
        segments.append(slice(int(start), int(start) + n_partial_frames + 1))
        start += n_partial_frames_f * (1 - overlap)
    return segments


def default_operator(x):
    return x


def sqrt_operator(x):
    return numpy.sqrt(x)


def default_normalize(x):
    return (x - x.min()) / (x.max() - x.min()) * 255


class InformativeImage(object):
    def __init__(self, frames):
        self.frames = frames
        self.informative_image = numpy.zeros_like(frames[0], dtype=float)

    def process(self):
        pass

    def save(self, filename):
        cv2.imwrite(filename, self.informative_image)


class MeanImage(InformativeImage):
    def __init__(self, frames):
        super(MeanImage, self).__init__(frames)
        self.T = len(self.frames)

    def process(self):
        for frame in self.frames:
            self.informative_image += 1 / self.T * frame
        return self


class MaxImage(InformativeImage):
    def __init__(self, frames):
        super(MaxImage, self).__init__(frames)
        self.sum = 0.0

    def process(self):
        self.frames = numpy.array(self.frames)
        self.informative_image = self.frames.max(axis=0)
        return self


class DynamicImage(InformativeImage):
    def __init__(self, frames):
        super(DynamicImage, self).__init__(frames)
        self.T = len(self.frames)

    def process(self, feature_operator=default_operator, get_coefficient=get_appr_coefficient_V,
                normalize=default_normalize):
        for t, frame in enumerate(self.frames, start=1):
            frame = feature_operator(frame)
            alpha = get_coefficient(self.T, t)
            self.informative_image += alpha * frame
        self.informative_image = normalize(self.informative_image)
        return self


class StaticImage(InformativeImage):
    def __init__(self, frames):
        super(StaticImage, self).__init__(frames)

    def process(self):
        self.informative_image = random.choice(self.frames)
        return self
