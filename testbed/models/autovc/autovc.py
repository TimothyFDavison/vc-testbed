# Model class for AutoVC system. #
import argparse
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
from . import autovc_fork


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
        Takes in .wav files as source, target.
        """
        # Format directory structure
        if os.path.isdir('wavs'):
            shutil.rmtree('wavs')
        os.mkdir('wavs')
        os.mkdir('wavs/s001')
        os.mkdir('wavs/t002')
        shutil.copyfile(source, f'wavs/s001/{source}')
        shutil.copyfile(target, f'wavs/t002/{dest}')

        # Generate spectrogram, speaker embeddings
        cmd = 'python3 autovc_fork/custom_make_spect.py'
        cmds = shlex.split(cmd)
        p = subprocess.Popen(cmds, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        cmd = 'python3 custom_make_metadata.py'
        cmds = shlex.split(cmd)
        p = subprocess.Popen(cmds, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        metadata = pickle.load(open('spmel/metadata.pkl', 'rb'))

        # Build conversion metadata file
        source_emb = source.replace(".wav", ".npy")
        source_embedding = f"spmel/s001/{source_emb}"
        dest_emb = target.replace(".wav", ".npy")
        dest_embedding = f"spmel/t002/{dest_emb}"
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
        model = autovc_fork.config.autovc_checkpoint
        def pad_seq(x, base=32):
            len_out = int(base * ceil(float(x.shape[0]) / base))
            len_pad = len_out - x.shape[0]
            assert len_pad >= 0
            return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

        device = 'cuda:0'
        G = autovc_fork.model_vc.Generator(32, 256, 512, 32).eval().to(device)
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

        target_pickle = 'conversion_results.pkl'
        with open(target_pickle, 'wb') as handle:
            pickle.dump(spect_vc, handle)
            print("Conversion success!")

        converted_spectrogram = "stub"
        return converted_spectrogram
