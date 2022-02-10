# Plug and play conversion script #
import argparse

from models.autovc.autovc import AutoVC


# Config
MODEL_MAP = {
    "autovc": AutoVC()
}


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
        "-o",
        "--outfile",
        dest="outfile",
        action="store",
        required=False,
        type=str,
        help="A specified output filepath."
    )
    return parser


def validate_args(args):
    """
    Validate that source/target files are available,
    """
    return


def preprocess_input(input, stage=None):
    """
    Preprocess input wav/ spectrogram.

    TODO: should we pass spectrograms or wav files to the conversion models?
          maybe each model's convert() function decides based on what is
          available? Though we should figure that out during the preprocess stage
          So maybe there's a config or mapping that determines input data types
          per model
    """
    processed_input = "stub"
    return processed_input


if __name__ == "__main__":
    # Interpret and validate CLI arguments.
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    # Instantiate and preprocess CLI arguments.
    model = MODEL_MAP[args.model]
    input_source = preprocess_input(args.source, stage="source")
    input_target = preprocess_input(args.target, stage="target")

    # Run conversion
    converted_sample = model.convert(
        input_source,
        input_target,
        outfile=args.outfile
    )

    # Wrap up
    print(f"Converted sample: {converted_sample}")
