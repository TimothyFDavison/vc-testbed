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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="CLI parser for conversion system inference."
    )
    parser.add_argument(
        "-s",
        "--source",
        dest="source",
        action="store",
        required=True,
        type=str,
        help="The source wav file or spectrogram for the conversion model."
    )
    parser.add_argument(
        "-t",
        "--target",
        dest="target",
        action="store",
        required=True,
        type=str,
        help="The target wav file or spectrogram for the conversion model."
    )
    parser.add_argument(
        "--output-prefix",
        dest="output_prefix",
        action="store",
        required=False,
        default="converted_",
        type=str,
        help="A specified output directory for the converted samples."
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        action="store",
        required=False,
        default=ROOT_DIR,
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

    # Instantiate and preprocess CLI arguments.
    for model_name, model in MODEL_MAP.items():
        print(f"Running {model_name}")
        setattr(args, "outfile_spect", os.path.join(args.out_dir, f"{args.output_prefix}{model_name}_spect"))
        setattr(args, "outfile_wav", os.path.join(args.out_dir, f"{args.output_prefix}{model_name}.wav"))

        print("Preprocessing...")
        input_source, input_target = model.preprocess_wavs(args.source, args.target)

        # Run conversion
        print("Running conversion...")
        converted_sample = model.convert(
            input_source,
            input_target,
            additional_args=args
        )
        if args.outfile_spect:
            np.save(args.outfile_spect, converted_sample)
        print("Done!")

        # Run signal reproduction
        if args.outfile_wav and not (model_name in VOCODES_BY_DEFAULT):
            print("Reconstructing signal...")
            waveform = model.vocode(converted_sample, outfile=args.outfile_wav)
            print("Done!")
