# Model class for AgainVC system. #
import torch

from ..model import ConversionSystem

# Module-specific imports
from . import (
    ADAINVC_CKPT,
    ADAINVC_VOCODER_CKPT
)
from .adainvc_fork import (
    inference
)


class AdainVC(ConversionSystem):
    """
    AgainVC wrapper class.
    """
    def __init__(self):
        return

    @staticmethod
    def preprocess_wavs(source, target):
        """
        Adjust the wav file to work with AgainVC
        Parameters:
            sampling rate: 16k
        """
        # Run AdainVC's preprocessing

        return source, target

    @staticmethod
    def convert(source, target, additional_args=None):
        """
        Run voice conversion over a provided source, target.
        Takes in .wav files as source, target.
        """
        if additional_args and additional_args.outfile_wav:
            outfile = additional_args.outfile_wav
        else:
            outfile = None
        converted_spectrogram = inference.main(
            ADAINVC_CKPT,
            ADAINVC_VOCODER_CKPT,
            source,
            target,
            outfile
        )[0]
        return converted_spectrogram

    @staticmethod
    def vocode(spectrogram, vocoder=None):
        return
