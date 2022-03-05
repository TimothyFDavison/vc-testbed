# Model class for AgainVC system. #
import librosa
import numpy as np
import soundfile as sf

from ..model import ConversionSystem

# Module-specific imports
from . import (
    STARGANV2VC_VOCODER_CKPT
)
from .starganv2vc_fork import (
    starganv2_lib
)


class StarGANv2VC(ConversionSystem):
    """
    StarGANv2VC wrapper class.
    """
    def __init__(self):
        return

    @staticmethod
    def preprocess_wavs(source, target):
        """
        Adjust the wav file to work with StarGANv2VC
        Parameters:
            sampling rate:
        """
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

        target_embedding = starganv2_lib.compute_speaker_embedding(target, None)

        # Preprocess source audio (from notebook)
        audio, source_sr = librosa.load(source, sr=24000)
        audio = audio / np.max(np.abs(audio))
        audio.dtype = np.float32
        source = starganv2_lib.preprocess(audio).to(starganv2_lib.device)

        # Run conversion
        f0_features = starganv2_lib.F0_model.get_feature_GAN(source.unsqueeze(1))
        out = starganv2_lib.starganv2.generator(source.unsqueeze(1), target_embedding, F0=f0_features)
        converted_spectrogram = out.transpose(-1, -2).squeeze().to(starganv2_lib.device).detach().cpu().numpy()

        if outfile:
            np.save(outfile, converted_spectrogram)
        return converted_spectrogram

    @staticmethod
    def vocode(spectrogram, vocoder=starganv2_lib.vocoder, outfile=None):
        output = vocoder.inference(spectrogram)
        output = output.view(-1).detach().cpu().numpy()

        # Optionally, save wav file
        if outfile:
           sf.write(outfile, output, 24000)
        return output

