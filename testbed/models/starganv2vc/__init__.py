# init file for testbed/models/starganv2vc #
import os

# AgainVC paths
STARGANV2VC_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.dirname(STARGANV2VC_DIR)
ROOT_DIR = os.path.dirname(MODELS_DIR)

STARGANV2VC_CKPT = f"{ROOT_DIR}/checkpoints/starganv2vc.pth"
STARGANV2VC_CKPT_CONFIG = f"{STARGANV2VC_DIR}/starganv2vc_fork/Models/config.yml"
STARGANV2VC_VOCODER_CKPT = f"{ROOT_DIR}/checkpoints/starganv2vc_vocoder.pkl"
STARGANV2VC_VOCODER_CKPT_CONFIG = f"{STARGANV2VC_DIR}/starganv2vc_fork/Vocoder/config.yml"
