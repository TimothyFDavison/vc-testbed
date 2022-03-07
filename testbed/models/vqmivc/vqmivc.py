# Model class for AutoVC system. #
from ..model import ConversionSystem

# Module-specific imports
from . import (
    VQMIVC_DIR,
    VQMIVC_CHECKPOINT,
    VQMIVC_VOCODER
)
from .vqmivc_fork import (
    convert_example
)


class VQMIVC(ConversionSystem):
    """
    AutoVC wrapper class.
    """
    def __init__(self):
        return

    @staticmethod
    def preprocess_wavs(source, target):
        """
        Adjust the wav file to work with VQMIVC
        Parameters:
            sampling rate:
        """
        outputs = [source, target]
        return tuple(outputs)

    @staticmethod
    def convert(source, target, additional_args=None):
        """
        Run voice conversion over a provided source, target.
        Takes in .wav files as source, target.
        """
        converted_spectrogram = convert_example.convert(
            source,
            target,
            VQMIVC_CHECKPOINT,
            VQMIVC_DIR,
            VQMIVC_VOCODER,
            additional_args
        )
        return converted_spectrogram

    @staticmethod
    def vocode(spectrogram, vocoder=None, outfile=None):
        """
        Reproduce audio signal.
        """
        return
