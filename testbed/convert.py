# Plug and play conversion script #
import argparse
import numpy as np
import soundfile as sf

from models.autovc.autovc import AutoVC
from models.againvc.againvc import AgainVC
from models.adainvc.adainvc import AdainVC
from models.starganv2vc.starganv2vc import StarGANv2VC

# Config
MODEL_MAP = {
    "autovc": AutoVC(),
    "againvc": AgainVC(),
    "adainvc": AdainVC(),
    "starganv2vc": StarGANv2VC()
}

VOCODES_BY_DEFAULT = [
    "adainvc",
    "againvc"
]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="CLI parser for conversion system inference."
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        action="store",
        required=True,
        type=str,
        help="The conversion model you intend to use."
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
        "--outfile-spect",
        dest="outfile_spect",
        action="store",
        required=False,
        type=str,
        help="A specified output filepath for the converted spectrogram."
    )
    parser.add_argument(
        "--outfile-wav",
        dest="outfile_wav",
        action="store",
        required=False,
        type=str,
        help="A specified output filepath for the converted wav file."
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
    model = MODEL_MAP[args.model]
    input_source, input_target = model.preprocess_wavs(args.source, args.target)

    # Run conversion
    converted_sample = model.convert(
        input_source,
        input_target,
        additional_args=args
    )
    if args.outfile_spect:
        np.save(args.outfile_spect, converted_sample)
    print(f"Converted sample: {converted_sample}")

    # Run signal reproduction
    if args.outfile_wav and not (args.model in VOCODES_BY_DEFAULT):
        waveform = model.vocode(converted_sample, outfile=args.outfile_wav)
        print(f"Converted sample (wav): {waveform}")
