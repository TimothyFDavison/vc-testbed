# Plug and play conversion script #
import argparse
import numpy as np
import os

from models import ROOT_DIR
from models.autovc.autovc import AutoVC
from models.againvc.againvc import AgainVC
from models.adainvc.adainvc import AdainVC
from models.starganv2vc.starganv2vc import StarGANv2VC
from models.vqmivc.vqmivc import VQMIVC

# Config
MODEL_MAP = {
    "againvc": AgainVC(),
    "adainvc": AdainVC(),
#    "starganv2vc": StarGANv2VC(),
    "vqmivc": VQMIVC(),
    "autovc": AutoVC(),
}

VOCODES_BY_DEFAULT = [
    "adainvc",
    "againvc",
    "vqmivc"
]

MODELS_TO_USE = ["adainvc", "vqmivc"]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="CLI parser for conversion system inference."
    )
    parser.add_argument(
        "-s",
        "--source_list",
        dest="source_list",
        action="store",
        required=True,
        type=str,
        help="A text file containing one source file's filepath per line."
    )
    parser.add_argument(
        "-t",
        "--target_list",
        dest="target_list",
        action="store",
        required=True,
        type=str,
        help="A text file containing one target file's filepath per line."
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        action="store",
        required=False,
        default="conversion_outputs",
        type=str,
        help="A specified output directory for the converted samples."
    )
    return parser


def validate_args(args):
    """
    Validate that source/target files are available,
    """
    return


if __name__ == "__main__":
    # Interpret and validate CLI arguments.
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    source_list = [x.strip() for x in open(args.source_list, "r")]
    target_list = [x.strip() for x in open(args.target_list, "r")]

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    for target_file in target_list:
        # Set up each target's directory
        target_name = target_file.strip().replace(".wav", "").split("/")[-1]
        target_dir = os.path.join(args.out_dir, target_name)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        else:
            continue  # temp - skipping over completed directories

        # Iterate through source files and convert
        for source_file in source_list:
            if source_file == target_file:
                continue
            source_name = source_file.strip().replace(".wav", "").split("/")[-1]
            converted_filename = os.path.join(target_dir, source_name + "_x_" + target_name)
            print("Generating", converted_filename, "...")

            setattr(args, "source", source_file)
            setattr(args, "target", target_file)

            for model_name in MODELS_TO_USE:

                setattr(args, "outfile_spect", f"{converted_filename}_spect_{model_name}.")
                setattr(args, "outfile_wav", f"{converted_filename}_{model_name}.wav")

                print(f"\t running {model_name}")
                model = MODEL_MAP[model_name]
                input_source, input_target = model.preprocess_wavs(args.source, args.target)
                converted_sample = model.convert(
                    input_source,
                    input_target,
                    additional_args=args
                )
                if args.outfile_wav and not (model_name in VOCODES_BY_DEFAULT):
                    waveform = model.vocode(converted_sample, outfile=args.outfile_wav)
