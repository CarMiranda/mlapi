import os
import tempfile
from starter import clean_data


def test__clean_data():

    _, filepath = tempfile.mkstemp(suffix="csv")
    with open(filepath, "w") as fp:
        fp.write(" " * 10000000)

    clean_data.remove_whitespaces(filepath, filepath)

    with open(filepath) as fp:
        text_content = fp.read()

    os.remove(filepath)

    assert (
        text_content == ""
    ), f"File should be empty, but actually contained: {text_content}"
