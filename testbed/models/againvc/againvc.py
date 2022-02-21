# Model class for AgainVC system. #
import argparse
import os
import pickle
from math import ceil
import numpy as np
import shlex
import shutil
import subprocess
import torch

from ..model import ConversionSystem

# Module-specific imports
from . import (
    AGAINVC_DIR,
    AGAINVC_CKPT,
    AGAINVC_INFERENCE_CONFIG,
    AGAINVC_PREPROCESS_CONFIG
)
from .againvc_fork import (
    agent,
    preprocessor,
    util
)


class AgainVCConfig:
    def __init__(self):
        return


class AgainVC(ConversionSystem):
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
        # Set up AgainVC's directory structure
        if os.path.isdir(f'{AGAINVC_DIR}/againvc_fork/data'):
            shutil.rmtree(f'{AGAINVC_DIR}/againvc_fork/data')
        os.mkdir(f'{AGAINVC_DIR}/againvc_fork/data')
        os.mkdir(f'{AGAINVC_DIR}/againvc_fork/data/s001')
        os.mkdir(f'{AGAINVC_DIR}/againvc_fork/data/t001')
        shutil.copyfile(source, f'{AGAINVC_DIR}/againvc_fork/data/s001/{source.replace("/", "_")}')
        shutil.copyfile(target, f'{AGAINVC_DIR}/againvc_fork/data/t001/{target.replace("/", "_")}')

        # Run AgainVC's preprocessing
        config = util.config.Config(AGAINVC_PREPROCESS_CONFIG)
        processor = preprocessor.get_preprocessor(config)
        for feat in config.feat_to_preprocess:
            processor.preprocess(
                input_path=config.input_path,
                output_path=config.output_path,
                feat=feat,
                njobs=4  # default
            )
        return source, target

    @staticmethod
    def convert(source, target, outfile=None):
        """
        Run voice conversion over a provided source, target.
        Takes in .wav files as source, target.
        """
        # Build config
        inf_config = util.config.Config(AGAINVC_INFERENCE_CONFIG)
        dsp_config = util.config.Config(AGAINVC_PREPROCESS_CONFIG)
        args = AgainVCConfig()
        output = f"{AGAINVC_DIR}/output"
        setattr(args, "dsp_config", dsp_config)
        setattr(args, "load", AGAINVC_CKPT)
        setattr(args, "source", source)
        setattr(args, "target", target)
        setattr(args, "output", output)
        setattr(args, "seglen", None)

        # Run inference
        model_inferencer = agent.inferencer.Inferencer(config=inf_config, args=args)
        model_inferencer.inference(
            source_path=args.source,
            target_path=args.target,
            out_path=args.output,
            seglen=args.seglen
        )

        converted_spectrogram = np.load(f'{AGAINVC_DIR}/output/mel/converted.npy')[0]
        converted_wav = ""  # see wav/ in output directory

        # Clean up artifacts
        shutil.rmtree(f'{AGAINVC_DIR}/againvc_fork/data')
        shutil.rmtree(f'{AGAINVC_DIR}/output')

        return converted_spectrogram
