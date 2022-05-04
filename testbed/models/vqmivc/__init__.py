# init file for testbed/models/autovc #
import os

VQMIVC_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.dirname(VQMIVC_DIR)
ROOT_DIR = os.path.dirname(MODELS_DIR)

VQMIVC_CHECKPOINT = f"{ROOT_DIR}/checkpoints/vqmivc_vctk.ckpt"
VQMIVC_VOCODER = f"{ROOT_DIR}/checkpoints/vqmivc_vocoder.pkl"
