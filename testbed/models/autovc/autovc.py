# Model class for AutoVC system. #
from ..model import ConversionSystem


class AutoVC(ConversionSystem):
    """
    AutoVC wrapper class.
    """
    def __init__(self):
        return

    @staticmethod
    def convert(source, target, outfile=None):
        """
        Run voice conversion over a provided source, target.
        """
        converted_spectrogram = "stub"
        return converted_spectrogram
