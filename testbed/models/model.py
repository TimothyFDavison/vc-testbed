# Abstract class for conversion systems #
from abc import ABC, abstractmethod


class ConversionSystem(ABC):
    """
    Conversion system
    """

    # Properties

    # Methods
    @abstractmethod
    def preprocess_wav(cls, wav):
        pass

    @abstractmethod
    def convert(cls, source, target):
        pass
