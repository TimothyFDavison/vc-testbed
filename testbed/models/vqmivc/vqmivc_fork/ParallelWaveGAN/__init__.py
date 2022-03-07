# init file for testbed/models/autovc #
import os

PARALLEL_WAVEGAN_DIR = os.path.dirname(os.path.realpath(__file__))
VQMIVC_FORK_DIR = os.path.dirname(PARALLEL_WAVEGAN_DIR)
VQMIVC_DIR = os.path.dirname(VQMIVC_FORK_DIR)
MODELS_DIR = os.path.dirname(VQMIVC_DIR)
ROOT_DIR = os.path.dirname(MODELS_DIR)
