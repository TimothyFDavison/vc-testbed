# init file for testbed/models/autovc #
import os

AUTOVC_FORK_DIR = os.path.dirname(os.path.realpath(__file__))
AUTOVC_DIR = os.path.dirname(AUTOVC_FORK_DIR)
MODELS_DIR = os.path.dirname(AUTOVC_DIR)
ROOT_DIR = os.path.dirname(MODELS_DIR)

AUTOVC_PYTHON = "/home/davistf1/testbed_venv/bin/python"
AUTOVC_VOCODER = f"{ROOT_DIR}/checkpoints/autovc_vocoder.pth"
