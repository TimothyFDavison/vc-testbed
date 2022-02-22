# init file for testbed/models/autovc #
import os

# AgainVC paths
AGAINVC_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.dirname(AGAINVC_DIR)
ROOT_DIR = os.path.dirname(MODELS_DIR)

AGAINVC_PYTHON = "/home/davistf1/testbed_venv/bin/python"
AGAINVC_CKPT = f"{ROOT_DIR}/checkpoints/againvc.pth"

# AgainVC model configs
AGAINVC_PREPROCESS_CONFIG = f"{AGAINVC_DIR}/againvc_fork/config/preprocess.yaml"
AGAINVC_INFERENCE_CONFIG = f"{AGAINVC_DIR}/againvc_fork/config/train_again-c4s.yaml"
