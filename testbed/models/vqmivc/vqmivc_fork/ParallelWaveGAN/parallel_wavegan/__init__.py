# init file for testbed/models/vqmivc #
import os

__version__ = "0.5.3"

PARALLEL_WAVEGAN_SUBDIR_BIN_DIR = os.path.dirname(os.path.realpath(__file__))
PARALLEL_WAVEGAN_SUBDIR = os.path.dirname(PARALLEL_WAVEGAN_SUBDIR_BIN_DIR)
PARALLEL_WAVEGAN_DIR = os.path.dirname(PARALLEL_WAVEGAN_SUBDIR)
VQMIVC_FORK_DIR = os.path.dirname(PARALLEL_WAVEGAN_DIR)
VQMIVC_DIR = os.path.dirname(VQMIVC_FORK_DIR)
MODELS_DIR = os.path.dirname(VQMIVC_DIR)
ROOT_DIR = os.path.dirname(MODELS_DIR)
