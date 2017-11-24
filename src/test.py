import logging
from typing import Set, List
import numpy as np
from numpy import ndarray
import os

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from src import config
from src.config import input_length
from src.detector import load_test_data_incorrectly
from src.neural_network_trainer import load_vocab, build_vocab_tokenizer_from_set, load_data, load_model, get_files, \
    encode_sentence, to_language, load_vocab_tokenizer, load_sentence

vocab: Set[str] = load_vocab(config.vocab_location)
vocab_tokenizer: Tokenizer = load_vocab_tokenizer(config.vocab_tokenizer_location)
# print(vocab_tokenizer.texts_to_sequences(["size1"]))

# x_test, y_test = load_data(f"{config.data_dir}/test", vocab, vocab_tokenizer)

total_count = 0
correct_count = 0
model = load_model(config.model_file_location, config.weights_file_location)


# x_test, y_test = load_data(os.path.join(config.data_dir, "test"), vocab, vocab_tokenizer)
x_test, correct_languages = load_test_data_incorrectly()
# correct_languages = [to_language(y) for y in y_test]
# x_test: ndarray = np.loadtxt("../resources/x_test.txt")

y_probas = model.predict(x_test)
predicted_languages = [to_language(y_proba) for y_proba in y_probas]
print(f"correct_language: {correct_languages}")
print(f"predicted_language: {predicted_languages}")
for correct_lang, predicted_lang in zip(correct_languages, predicted_languages):
    if correct_lang == predicted_lang:
        correct_count += 1
    total_count += 1
print(f"accuracy: {correct_count / total_count}")
