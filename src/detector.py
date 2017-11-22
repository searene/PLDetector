from typing import Dict, List, Iterable
import os

from keras import Sequential
from keras.models import model_from_json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from src import config
from src.config import input_length
from src.github_fetcher import ext_lang_dict
from src.neural_network_trainer import build_vocab_tokenizer_from_file, encode_sentence, to_language, \
    NeuralNetworkTrainer, load_model, to_binary_list

vocab_tokenizer: Tokenizer = build_vocab_tokenizer_from_file(config.vocab_location)


def get_neural_network_input(code: str) -> np.ndarray:
    encoded_sentence: List[int] = encode_sentence(code, vocab_tokenizer)
    return pad_sequences([encoded_sentence], maxlen=input_length)


def detect(code: str, model=None):
    if model is None:
        model = load_model(config.model_file_location, config.weights_file_location)
    y_proba = model.predict(get_neural_network_input(code))
    return to_language(y_proba)


def calculate_accuracy():
    total_count = 0
    correct_count = 0
    neural_network_trainer = NeuralNetworkTrainer(
        f"{config.data_dir}/train",
        f"{config.data_dir}/test",
        config.vocab_location,
        config.word2vec_location,
        ext_lang_dict)
    files: List[str] = neural_network_trainer.get_files([os.path.join(config.data_dir, "test")])
    for f in files:
        with open(f, "r") as opened_file:
            file_contents = opened_file.read()
        correct_language: str = os.path.dirname(f).split(os.path.sep)[-1]
        predicted_language = detect(file_contents)
        print(f"correct_language: {correct_language}")
        print(f"predicted_language: {predicted_language}")
        total_count += 1
        if correct_language == predicted_language:
            correct_count += 1
    print(f"accuracy: {correct_count / total_count}")


def convert_sigmoid_result_to_binary_list(sigmoid_result: Iterable[List[int]]) -> List[List[int]]:
    result: List[List[int]] = []
    for row in sigmoid_result:
        converted_row: List[int] = to_binary_list(np.argmax(row), len(row))
        result.append(converted_row)
    return result


def load_saved_training_data_and_evaluate():
    total_count = 0
    correct_count = 0
    model = load_model("../resources/models/model.json", "../resources/models/model.h5")
    x_test: np.ndarray = np.loadtxt("../resources/x_test.txt")
    y_test: np.ndarray = np.loadtxt("../resources/y_test.txt")
    y_predicted: np.ndarray = model.predict(x_test)

    y_correct_list = [[int(v) for v in value] for value in y_test.tolist()]
    y_predicted_list = convert_sigmoid_result_to_binary_list(y_predicted.tolist())

    y_correct_languages = [to_language(binary_list) for binary_list in y_correct_list]
    y_predicted_languages = [to_language(binary_list) for binary_list in y_predicted_list]

    print(f"y_correct_list: {y_correct_list}")
    print(f"y_predicted_list: {y_predicted_list}")
    print(f"y_correct_languages: {y_correct_languages}")
    print(f"y_predicted_languages: {y_predicted_languages}")

    for correct_lang, predicted_lang in zip(y_correct_languages, y_predicted_languages):
        if correct_lang == predicted_lang:
            correct_count += 1
        total_count += 1
    print(f"accuracy: {correct_count / total_count}")


if __name__ == "__main__":
    # calculate_accuracy()
    load_saved_training_data_and_evaluate()
