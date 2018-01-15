import numpy
from math import log

class HarmonicSequence(object):
    __Harmonic_table__ = None
    gamma = 0.57721566490153286060651209008240243104215933593992

    @staticmethod
    def at(index):
        if HarmonicSequence.__Harmonic_table__ is None:
            HarmonicSequence.__Harmonic_table__ = HarmonicSequence.calculate_harmonic_table()
        if index < 0:
            raise IndexError("The index of Harmonic sequence can not be negative.")
        elif index < 1000:
            return HarmonicSequence.__Harmonic_table__[index]
        else:
            return HarmonicSequence.gamma + log(index) + 0.5 / index - 1. / (12 * index ** 2) + 1. / (120 * index ** 4)

    @staticmethod
    def calculate_harmonic_table():
        table = numpy.empty(shape=(1000), dtype=float)
        table[0] = 0
        for t in range(1, 1000):
            table[t] = 1 / t + table[t - 1]
        return table