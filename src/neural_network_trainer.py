import logging
import re
from typing import List, Set, Counter, Union, Dict
import numpy as np
import os
from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import asarray, ndarray, zeros
from src import config
from src.config import input_length
from src.github_fetcher import ext_lang_dict
import pickle


def clean_vocab_and_word2vec():
    try:
        os.remove(config.vocab_location)
    except FileNotFoundError:
        pass
    try:
        os.remove(config.vocab_tokenizer_location)
    except FileNotFoundError:
        pass
    try:
        os.remove(config.word2vec_location)
    except FileNotFoundError:
        pass


def save_vocab_tokenizer(vocab_tokenzier_location, vocab_tokenizer):
    with open(vocab_tokenzier_location, 'wb') as f:
        pickle.dump(vocab_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_vocab_tokenizer(vocab_tokenizer_location):
    with open(vocab_tokenizer_location, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def evaluate_saved_data(x_file_name, y_file_name, model):
    x = np.loadtxt(x_file_name)
    y = np.loadtxt(y_file_name)
    loss, accuracy = model.evaluate(x, y, verbose=2)
    print(f"loss: {loss}, accuracy: {accuracy}")


def get_all_languages():
    result = []
    for value in ext_lang_dict.values():
        if type(value) is list:
            result.extend(value)
        else:
            result.append(value)
    return result


def to_binary_list(i, count):
    result = [0] * count
    result[i] = 1
    return result


def to_language(binary_list):
    languages = get_all_languages()
    i = np.argmax(binary_list)
    return languages[i]


def get_lang_sequence(lang):
    languages = get_all_languages()
    for i in range(len(languages)):
        if languages[i] == lang:
            return to_binary_list(i, len(languages))
    raise Exception(f"Language {lang} is not supported.")


def encode_sentence(sentence, vocab_tokenizer):
    encoded_sentence = vocab_tokenizer.texts_to_sequences(sentence.split())
    return [word[0] for word in encoded_sentence if len(word) != 0]


def load_vocab(vocab_location):
    with open(vocab_location) as f:
        words = f.read().splitlines()
    return set(words)


def load_word2vec(word2vec_location):
    result = dict()
    with open(word2vec_location, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.split()
        result[parts[0]] = asarray(parts[1:], dtype="float32")
    return result


def load_model(model_file_location, weights_file_location):
    with open(model_file_location) as f:
        model = model_from_json(f.read())
    model.load_weights(weights_file_location)
    return model


def build_vocab_tokenizer_from_set(s):
    vocab_tokenizer = Tokenizer(lower=False, filters="")
    vocab_tokenizer.fit_on_texts(s)
    return vocab_tokenizer


def build_vocab_tokenizer_from_file(vocab_location):
    s = load_vocab(vocab_location)
    return build_vocab_tokenizer_from_set(s)


def should_language_be_loaded(language):
    for value in ext_lang_dict.values():
        if value == language:
            return True
    return False


def get_files(data_dirs):
    result = []
    for data_dir in data_dirs:
        depth = 0
        for root, sub_folders, files in os.walk(data_dir):
            depth += 1

            # ignore the first loop
            if depth == 1:
                continue

            language = os.path.basename(root)
            if should_language_be_loaded(language):
                result.extend([os.path.join(root, f) for f in files])
            depth += 1
    return result


def load_words_from_str(s):
    contents = " ".join(s.splitlines())
    result = re.split(r"[{}()\[\]\'\":.*\s,#=_/\\><;?\-|+]", contents)

    # remove empty elements
    result = [word for word in result if word.strip() != ""]

    return result


def load_words_from_file(file_name):
    try:
        with open(file_name, "r") as f:
            contents = f.read()
    except UnicodeDecodeError:
        logging.warning(f"Encountered UnicodeDecodeError, ignore file {file_name}.")
        return []
    return load_words_from_str(contents)


def get_languages(ext_lang_dict):
    languages = set()
    for ext, language in ext_lang_dict.items():
        if type(language) is str:
            languages.update([language])
        elif type(language) is list:
            languages.update(language)
    return languages


def save_model(model, model_file_location, weights_file_location):
    with open(model_file_location, "w") as f:
        f.write(model.to_json())
    model.save_weights(weights_file_location)


def save_vocabulary(vocabulary, file_location):
    with open(file_location, "w+") as f:
        for word in vocabulary:
            f.write(word + "\n")


def load_sentence(file_name, vocab):
    """ Used in loading data, word that is not in the vocabulary will not be included
    """
    words = load_words_from_file(file_name)
    return " ".join([word for word in words if word in vocab])


def load_data(data_dir, vocab, vocab_tokenizer):
    files = get_files([data_dir])
    x = []
    y = []
    for f in files:
        language = os.path.dirname(f).split(os.path.sep)[-1]
        sentence = load_sentence(f, vocab)
        x.append(encode_sentence(sentence, vocab_tokenizer))
        y.append(get_lang_sequence(language))
    return pad_sequences(x, maxlen=input_length), asarray(y)


class NeuralNetworkTrainer:
    def __init__(self, train_data_dir, test_data_dir, vocab_location, vocab_tokenizer_location, word2vec_location,
                 ext_lang_dict):
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.vocab_location = vocab_location
        self.word2vec_location = word2vec_location
        self.vocab_tokenizer = None
        self.language_tokenizer = None
        self.wordvec_dimension = 100
        self.ext_lang_dict = ext_lang_dict

        languages = get_languages(ext_lang_dict)
        self.__build_language_tokenizer(list(languages))

        if not os.path.exists(self.vocab_location):
            self.build_and_save_vocabulary()
        if not os.path.exists(self.word2vec_location):
            self.build_and_save_word2vec_model()

        self.word2vec = load_word2vec(self.word2vec_location)
        self.vocab = load_vocab(self.vocab_location)

        if not os.path.exists(vocab_tokenizer_location):
            vocab_tokenizer = build_vocab_tokenizer_from_set(self.vocab)
            save_vocab_tokenizer(vocab_tokenizer_location, vocab_tokenizer)

        self.vocab_tokenizer = load_vocab_tokenizer(vocab_tokenizer_location)

    def build_vocabulary(self):
        vocabulary = Counter()
        files = get_files([self.train_data_dir, self.test_data_dir])
        for f in files:
            words = load_words_from_file(f)
            vocabulary.update(words)

        # remove rare words
        min_count = 5
        vocabulary = [word for word, count in vocabulary.items() if count >= min_count]
        return vocabulary

    def build_and_save_vocabulary(self):
        vocabulary = self.build_vocabulary()
        save_vocabulary(vocabulary, self.vocab_location)

    def build_word2vec_model(self, vocabulary):
        all_tokens = []
        files = get_files([self.train_data_dir, self.test_data_dir])
        for f in files:
            words = load_words_from_file(f)
            all_tokens.append([token for token in words if token in vocabulary])
        logging.info("Building the Word2Vec model...")
        model = Word2Vec(all_tokens, size=100, window=5, workers=8, min_count=1)
        return model

    def build_and_save_word2vec_model(self):
        vocab = load_vocab(self.vocab_location)
        model = self.build_word2vec_model(vocab)
        logging.info("Saving the word2vec model to the disk...")
        model.wv.save_word2vec_format(self.word2vec_location, binary=False)

    def build_model(self):
        weight_matrix = self.__get_weights()

        # build the embedding layer
        input_dim = len(self.vocab_tokenizer.word_index) + 1
        output_dim = self.wordvec_dimension
        x_train, y_train = load_data(self.train_data_dir, self.vocab, self.vocab_tokenizer)

        embedding_layer = Embedding(input_dim, output_dim, weights=[weight_matrix], input_length=input_length,
                                    trainable=False)
        logging.info(f"x_train.shape: {x_train.shape}")
        logging.info(f"y_train.shape: {y_train.shape}")
        logging.info(f"weight_matrix.shape: {weight_matrix.shape}")

        model = Sequential()
        model.add(embedding_layer)
        model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(len(get_all_languages()), activation="sigmoid"))
        logging.info(model.summary())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, verbose=2)
        return model

    def evaluate_model(self, model):
        x_test, y_test = load_data(self.test_data_dir, self.vocab, self.vocab_tokenizer)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        logging.info('Test Accuracy: %f' % (acc * 100))

    def __build_language_tokenizer(self, languages):
        if self.language_tokenizer is None:
            self.language_tokenizer = Tokenizer()
            self.language_tokenizer.fit_on_texts(languages)
        return self.language_tokenizer

    def __get_weights(self):
        vocab_size = len(self.vocab_tokenizer.word_index) + 1
        weight_matrix = zeros((vocab_size, self.wordvec_dimension))
        for word, index in self.vocab_tokenizer.word_index.items():
            weight_matrix[index] = self.word2vec[word]
        return weight_matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # clean_vocab_and_word2vec()
    neural_network_trainer = NeuralNetworkTrainer(
        f"{config.data_dir}/train",
        f"{config.data_dir}/test",
        config.vocab_location,
        config.vocab_tokenizer_location,
        config.word2vec_location,
        ext_lang_dict)
    model = neural_network_trainer.build_model()
    neural_network_trainer.evaluate_model(model)
    save_model(model, config.model_file_location, config.weights_file_location)
