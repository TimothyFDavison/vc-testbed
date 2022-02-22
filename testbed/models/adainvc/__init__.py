# init file for testbed/models/autovc #
import os

# AgainVC paths
ADAINVC_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.dirname(ADAINVC_DIR)
ROOT_DIR = os.path.dirname(MODELS_DIR)

ADAINVC_PYTHON = "/home/davistf1/testbed_venv/bin/python"
ADAINVC_CKPT = f"{ROOT_DIR}/checkpoints/adainvc.ckpt"
ADAINVC_VOCODER_CKPT = f"{ROOT_DIR}/checkpoints/adainvc_vocoder.pt"
