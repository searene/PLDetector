import logging
from typing import Set, List
import numpy as np
import os

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from src import config
from src.config import input_length
from src.neural_network_trainer import load_vocab, build_vocab_tokenizer_from_set, load_data, load_model, get_files, \
    encode_sentence, to_language, load_vocab_tokenizer

vocab: Set[str] = load_vocab(config.vocab_location)
vocab_tokenizer: Tokenizer = load_vocab_tokenizer(config.vocab_tokenizer_location)
print(vocab_tokenizer.texts_to_sequences(["size1"]))

x_test, y_test = load_data(f"{config.data_dir}/test", vocab, vocab_tokenizer)

total_count = 0
correct_count = 0
model = load_model(config.model_file_location, config.weights_file_location)

def load_test_data_incorrectly():
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

    encoded_sentences: List[List[int]] = [encode_sentence(code, vocab_tokenizer) for code in file_contents_list]
    x_test = pad_sequences(encoded_sentences, maxlen=input_length)
    y_correct: List[str] = [os.path.dirname(f).split(os.path.sep)[-1] for f in processed_files]
    return x_test, y_correct

_, correct_languages = load_test_data_incorrectly()
x_test: np.ndarray = np.loadtxt("../resources/x_test.txt")

y_probas = model.predict(x_test)
predicted_languages = [to_language(y_proba) for y_proba in y_probas]
print(f"correct_language: {correct_languages}")
print(f"predicted_language: {predicted_languages}")
for correct_lang, predicted_lang in zip(correct_languages, predicted_languages):
    if correct_lang == predicted_lang:
        correct_count += 1
    total_count += 1
print(f"accuracy: {correct_count / total_count}")
