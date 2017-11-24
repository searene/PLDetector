import logging
from typing import Dict, List, Iterable, Set
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
    NeuralNetworkTrainer, load_model, to_binary_list, get_files, load_vocab, build_vocab_tokenizer_from_set, load_data, \
    save_numpy_arrays, load_vocab_tokenizer, load_sentence

vocab: Set[str] = load_vocab(config.vocab_location)
vocab_tokenizer: Tokenizer = load_vocab_tokenizer(config.vocab_tokenizer_location)

x_test, y_test = load_data(f"{config.data_dir}/test", vocab, vocab_tokenizer)

# save_numpy_arrays({
#     "../resources/x_test.txt": x_test,
#     "../resources/y_test.txt": y_test
# })


def get_neural_network_input(code: str) -> np.ndarray:
    filtered_sentence: str = " ".join([word for word in code if word in vocab])
    encoded_sentence: List[int] = encode_sentence(filtered_sentence, vocab_tokenizer)
    return pad_sequences([encoded_sentence], maxlen=input_length)


def get_batch_neural_network_input(codes: List[str]) -> np.ndarray:
    encoded_sentences: List[List[int]] = [encode_sentence(code, vocab_tokenizer) for code in codes]
    padded_sentences = pad_sequences(encoded_sentences, maxlen=input_length)
    return padded_sentences


def detect(code: str, model=None):
    if model is None:
        model = load_model(config.model_file_location, config.weights_file_location)
    y_proba = model.predict(get_neural_network_input(code))
    return to_language(y_proba)


def batch_detect(codes: List[str], model=None):
    if model is None:
        model = load_model(config.model_file_location, config.weights_file_location)
    y_probas = model.predict(get_batch_neural_network_input(codes))
    return [to_language(y_proba) for y_proba in y_probas]


def load_test_data_incorrectly():
    files: List[str] = get_files([os.path.join(config.data_dir, "test")])
    processed_files: List[str] = []
    sentence_list: List[str] = []
    for f in files:
        sentence = load_sentence(f, vocab)
        if len(sentence) == 0:
            continue
        sentence_list.append(sentence)
        processed_files.append(f)

    encoded_sentences: List[List[int]] = [encode_sentence(sentence, vocab_tokenizer) for sentence in sentence_list]
    x_test = pad_sequences(encoded_sentences, maxlen=input_length)
    y_correct: List[str] = [os.path.dirname(f).split(os.path.sep)[-1] for f in processed_files]
    return x_test, y_correct


def calculate_accuracy():
    files: List[str] = get_files([os.path.join(config.data_dir, "test")])
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
    calculate_accuracy()
    # load_saved_training_data_and_evaluate()
