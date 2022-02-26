# Script to clean the raw data

import argparse
import pathlib

def get_args():
    """Parses CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", type=pathlib.Path, help="Path to input file (raw data)")
    parser.add_argument("-o", "--outfile", type=pathlib.Path, help="Path to output file (clean data)")

    return parser.parse_args()

def remove_whitespaces(infile: pathlib.Path, outfile: pathlib.Path) -> None:
    """Removes all whitespaces from a given file and writes the result to a new file.

    Args:
        infile (pathlib.Path): Input file path
        outfile (pathlib.Path): Output file path
    """

    with open(infile) as fp:
        text_content = fp.read()

    text_content = text_content.replace(" ", "")

    with open(outfile, "w") as fp:
        fp.write(text_content)

if __name__ == "__main__":
    args = get_args()
    remove_whitespaces(args.infile, args.outfile)
