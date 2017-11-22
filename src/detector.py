from typing import Dict, List
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
    NeuralNetworkTrainer

vocab_tokenizer: Tokenizer = build_vocab_tokenizer_from_file(config.vocab_location)

def get_seq_to_lang_dict():
    result = {}
    lang_to_seq_dict = get_lang_to_seq_dict(ext_lang_dict)
    for key, value in lang_to_seq_dict.items():
        result[value] = key
    return result


def load_model(model_file_location: str, weights_file_location: str) -> Sequential:
    with open(model_file_location) as f:
        model = model_from_json(f.read())
    model.load_weights(weights_file_location)
    return model


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


if __name__ == "__main__":
    calculate_accuracy()
