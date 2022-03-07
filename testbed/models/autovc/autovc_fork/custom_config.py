# Configuration file for AutoVC model #
import os

AUTOVC_FORK_DIR = os.path.dirname(os.path.realpath(__file__))
AUTOVC_DIR = os.path.dirname(AUTOVC_FORK_DIR)
MODELS_DIR = os.path.dirname(AUTOVC_DIR)
ROOT_DIR = os.path.dirname(MODELS_DIR)

# Model checkpoints
autovc_checkpoint = f"{ROOT_DIR}/checkpoints/autovc.ckpt"
speaker_encoder_checkpoint = f"{ROOT_DIR}/checkpoints/autovc_encoder.ckpt"
wavenet_checkpoint = None
