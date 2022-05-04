# Model class for AutoVC system. #
import os
import pickle
from math import ceil
import numpy as np
import shlex
import shutil
import soundfile as sf
import subprocess
import torch

from ..model import ConversionSystem

# Module-specific imports
from . import (
    AUTOVC_DIR,
    AUTOVC_PYTHON,
    AUTOVC_VOCODER
)
from .autovc_fork import (
    custom_config,
    model_vc,
    synthesis
)


class AutoVC(ConversionSystem):
    """
    AutoVC wrapper class.
    """
    def __init__(self):
        return

    @staticmethod
    def preprocess_wavs(source, target):
        """
        Adjust the wav file to work with AutoVC
        Parameters:
            sampling rate: 16k
        """
        outputs = []
        for wav in [source, target]:
            output_wav = f"{wav}.autovc.wav"
            adjust_sampling_rate = f"ffmpeg -y -i {wav} -ar 16000 {output_wav}"
            os.system(adjust_sampling_rate)  # TODO: build out utils.py file with CLI runner (subprocess.POpen)
            outputs.append(output_wav)

        return tuple(outputs)

    @staticmethod
    def convert(source, target, additional_args=None):
        """
        Run voice conversion over a provided source, target.
        Takes in .wav files as source, target.
        """
        # Format directory structure
        if os.path.isdir(f'{AUTOVC_DIR}/wavs'):
            shutil.rmtree(f'{AUTOVC_DIR}/wavs')
        os.mkdir(f'{AUTOVC_DIR}/wavs')
        os.mkdir(f'{AUTOVC_DIR}/wavs/s001')
        os.mkdir(f'{AUTOVC_DIR}/wavs/t002')
        shutil.copyfile(source, f'{AUTOVC_DIR}/wavs/s001/{source.replace("/", "_")}')
        shutil.copyfile(target, f'{AUTOVC_DIR}/wavs/t002/{target.replace("/", "_")}')
        os.remove(source)
        os.remove(target)

        # Generate spectrogram, speaker embeddings
        os.chdir(AUTOVC_DIR)
        cmd = f'{AUTOVC_PYTHON} {AUTOVC_DIR}/autovc_fork/custom_make_spect.py'
        cmds = shlex.split(cmd)
        p = subprocess.Popen(cmds, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        cmd = f'{AUTOVC_PYTHON} {AUTOVC_DIR}/autovc_fork/custom_make_metadata.py'
        cmds = shlex.split(cmd)
        p = subprocess.Popen(cmds, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        metadata = pickle.load(open(f'{AUTOVC_DIR}/spmel/metadata.pkl', 'rb'))

        # Build conversion metadata file
        source_emb = source.replace("/", "_")[:-3] + "npy"
        source_embedding = f"{AUTOVC_DIR}/spmel/s001/{source_emb}"
        dest_emb = target.replace("/", "_")[:-3] + "npy"
        dest_embedding = f"{AUTOVC_DIR}/spmel/t002/{dest_emb}"
        source_metadata = [
            "s001",
            metadata[0][1],
            np.load(source_embedding)
        ]
        target_metadata = [
            "t002",
            metadata[1][1],
            np.load(dest_embedding)
        ]
        conversion_metadata = [source_metadata, target_metadata]

        # Run conversion
        model = custom_config.autovc_checkpoint
        def pad_seq(x, base=32):
            len_out = int(base * ceil(float(x.shape[0]) / base))
            len_pad = len_out - x.shape[0]
            assert len_pad >= 0
            return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

        device = 'cuda:0'
        G = model_vc.Generator(16, 256, 512, 32).eval().to(device)  # first param 32 for pretrained model
        g_checkpoint = torch.load(model, map_location='cuda:0')
        G.load_state_dict(g_checkpoint['model'])

        spect_vc = []
        for sbmt_i in conversion_metadata:
            x_org = sbmt_i[2]
            x_org, len_pad = pad_seq(x_org)
            uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
            emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
            for sbmt_j in conversion_metadata:
                if sbmt_i[0] == sbmt_j[0]:  # skipping reflexive conversion
                    continue
                elif sbmt_i[0] == "t002":  # skipping target -> source conversion
                    continue
                emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
                with torch.no_grad():
                    _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
                if len_pad == 0:
                    uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
                else:
                    uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
                spect_vc.append(('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg))

        # Clean up intermediary files
        shutil.rmtree(f'{AUTOVC_DIR}/wavs')
        shutil.rmtree(f'{AUTOVC_DIR}/spmel')

        converted_spectrogram = spect_vc[0][1]
        return converted_spectrogram

    @staticmethod
    def vocode(spectrogram, vocoder=AUTOVC_VOCODER, outfile=None):
        """
        Reproduce audio signal.
        """
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        model = synthesis.build_model().to(device)
        checkpoint = torch.load(vocoder)
        model.load_state_dict(checkpoint["state_dict"])
        waveform = synthesis.wavegen(model, spectrogram)

        if outfile:
            sf.write(outfile, waveform, 16000)
        return waveform
