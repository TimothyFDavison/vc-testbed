# Abstract class for conversion systems #
from abc import ABC, abstractmethod


class ConversionSystem(ABC):
    """
    Conversion system
    """

    # Properties

    # Methods
    @abstractmethod
    def preprocess_wavs(cls, source, target):
        pass

    @abstractmethod
    def convert(cls, source, target):
        pass

    @abstractmethod
    def vocode(cls, spectrogram, vocoder=None):
        pass
