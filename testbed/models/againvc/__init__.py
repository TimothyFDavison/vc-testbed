# init file for testbed/models/autovc #
import os

# AgainVC paths
AGAINVC_DIR = os.path.dirname(os.path.realpath(__file__))
AGAINVC_PYTHON = "/home/davistf1/testbed_venv/bin/python"
print(os.getcwd())
AGAINVC_CKPT = f"testbed/checkpoints/againvc.pth"

# AgainVC model configs
AGAINVC_PREPROCESS_CONFIG = f"{AGAINVC_DIR}/againvc_fork/config/preprocess.yaml"
AGAINVC_INFERENCE_CONFIG = f"{AGAINVC_DIR}/againvc_fork/config/train_again-c4s.yaml"
