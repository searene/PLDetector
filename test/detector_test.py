import logging
from typing import List

import os

from src import config
from src.detector import detect
from src.neural_network_trainer import get_files


def calculate_accuracy():
    files: List[str] = get_files(os.path.join(config.data_dir, "test"))
    processed_files: List[str] = []
    file_contents_list: List[str] = []
    for f in files:
        try:
            with open(f, "r") as opened_file:
                file_contents = opened_file.read()
                file_contents_list.append(file_contents)
                processed_files.append(f)
        except UnicodeDecodeError:
            logging.error(f"Error occurred while reading {f}")

    total_count = 0
    correct_count = 0
    correct_languages = [os.path.dirname(f).split(os.path.sep)[-1] for f in processed_files]
    predicted_languages = [detect(file_contents) for file_contents in file_contents_list]
    for correct_lang, predicted_lang in zip(correct_languages, predicted_languages):
        if correct_lang == predicted_lang:
            correct_count += 1
        total_count += 1
    print(f"accuracy: {correct_count / total_count}")


if __name__ == '__main__':
    calculate_accuracy()
