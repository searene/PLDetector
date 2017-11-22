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


def evaluate_saved_data(x_file_name: str, y_file_name: str, model: Sequential) -> None:
    x: ndarray = np.loadtxt(x_file_name)
    y: ndarray = np.loadtxt(y_file_name)
    loss, accuracy = model.evaluate(x, y, verbose=2)
    print(f"loss: {loss}, accuracy: {accuracy}")

def get_all_languages() -> List[str]:
    result: List[str] = []
    for value in ext_lang_dict.values():
        if type(value) is list:
            result.extend(value)
        else:
            result.append(value)
    return result


def to_binary_list(i: int, count: int) -> List[int]:
    result = [0] * count
    result[i] = 1
    return result


def to_language(binary_list: List[int]) -> str:
    languages: List[str] = get_all_languages()
    i: int = np.argmax(binary_list)
    return languages[i]


def get_lang_sequence(lang: str) -> List[int]:
    languages = get_all_languages()
    for i in range(len(languages)):
        if languages[i] == lang:
            return to_binary_list(i, len(languages))
    raise Exception(f"Language {lang} is not supported.")


def encode_sentence(sentence: str, vocab_tokenizer: Tokenizer) -> List[int]:
    encoded_sentence: List[List[int]] = vocab_tokenizer.texts_to_sequences(sentence.split())
    return [word[0] for word in encoded_sentence if len(word) != 0]


def load_vocab(vocab_location) -> Set[str]:
    with open(vocab_location) as f:
        words = f.read().splitlines()
    return set(words)


def load_word2vec(word2vec_location: str) -> Dict[str, ndarray]:
    result: Dict[str, ndarray] = dict()
    with open(word2vec_location, "r", encoding="utf-8") as f:
        lines: List[str] = f.readlines()[1:]
    for line in lines:
        parts: List[str] = line.split()
        result[parts[0]] = asarray(parts[1:], dtype="float32")
    return result


def load_model(model_file_location: str, weights_file_location: str) -> Sequential:
    with open(model_file_location) as f:
        model = model_from_json(f.read())
    model.load_weights(weights_file_location)
    return model


def build_vocab_tokenizer_from_set(s: Set[str]) -> Tokenizer:
    vocab_tokenizer = Tokenizer(lower=False, filters="")
    vocab_tokenizer.fit_on_texts(s)
    return vocab_tokenizer


def build_vocab_tokenizer_from_file(vocab_location: str) -> Tokenizer:
    s: Set[str] = load_vocab(vocab_location)
    return build_vocab_tokenizer_from_set(s)


def save_numpy_arrays(numpy_arrays: Dict[str, ndarray]):
    for file_name, array in numpy_arrays.items():
        np.savetxt(file_name, array)


class NeuralNetworkTrainer:
    def __init__(self, train_data_dir: str, test_data_dir: str, vocab_location: str, word2vec_location: str,
                 ext_lang_dict: Dict[str, Union[str, List[str]]]):
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.vocab_location = vocab_location
        self.word2vec_location = word2vec_location
        self.vocab_tokenizer: Tokenizer = None
        self.language_tokenizer: Tokenizer = None
        self.wordvec_dimension = 100
        self.ext_lang_dict = ext_lang_dict

        languages = self.__get_languages(ext_lang_dict)
        self.__build_language_tokenizer(list(languages))

        self.build_and_save_vocabulary()
        self.build_and_save_word2vec_model()

        self.word2vec: Dict[str, ndarray] = load_word2vec(self.word2vec_location)
        self.vocab: Set[str] = load_vocab(self.vocab_location)
        self.vocab_tokenizer: Tokenizer = build_vocab_tokenizer_from_set(self.vocab)

    def build_vocabulary(self) -> Counter:
        vocabulary: Counter = Counter()
        files = self.get_files([self.train_data_dir, self.test_data_dir])
        for f in files:
            words = self.__load_words_from_file(f)
            vocabulary.update(words)

        # remove rare words
        min_count = 5
        vocabulary = [word for word, count in vocabulary.items() if count >= min_count]
        return vocabulary

    def save_vocabulary(self, vocabulary: Counter, file_location: str) -> None:
        with open(file_location, "w+") as f:
            for word in vocabulary:
                f.write(word + "\n")

    def build_and_save_vocabulary(self) -> None:
        vocabulary: Counter = self.build_vocabulary()
        self.save_vocabulary(vocabulary, self.vocab_location)

    def build_word2vec_model(self, vocabulary: Set[str]) -> Word2Vec:
        all_tokens: List[List[str]] = []
        files: List[str] = self.get_files([self.train_data_dir, self.test_data_dir])
        for f in files:
            words: List[str] = self.__load_words_from_file(f)
            all_tokens.append([token for token in words if token in vocabulary])
        logging.info("Building the Word2Vec model...")
        model = Word2Vec(all_tokens, size=100, window=5, workers=8, min_count=1)
        return model

    def build_and_save_word2vec_model(self) -> None:
        vocab = load_vocab(self.vocab_location)
        model = self.build_word2vec_model(vocab)
        logging.info("Saving the word2vec model to the disk...")
        model.wv.save_word2vec_format(self.word2vec_location, binary=False)

    def build_model(self) -> Sequential:
        weight_matrix: ndarray = self.__get_weights()

        # build the embedding layer
        input_dim = len(self.vocab_tokenizer.word_index) + 1
        output_dim = self.wordvec_dimension
        x_train, y_train = self.load_data(self.train_data_dir)

        save_numpy_arrays({
            "../resources/x_train.txt": x_train,
            "../resources/y_train.txt": y_train
        })

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

    def evaluate_model(self, model: Sequential) -> None:
        x_test, y_test = self.load_data(self.test_data_dir)
        save_numpy_arrays({
            "../resources/x_test.txt": x_test,
            "../resources/y_test.txt": y_test
        })
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        logging.info('Test Accuracy: %f' % (acc * 100))

    def save_model(self, model: Sequential, model_file_location, weights_file_location) -> None:
        with open(model_file_location, "w") as f:
            f.write(model.to_json())
        model.save_weights(weights_file_location)

    def __get_languages(self, ext_lang_dict: Dict[str, Union[str, List[str]]]) -> Set[str]:
        languages: Set[str] = set()
        for ext, language in ext_lang_dict.items():
            if type(language) is str:
                languages.update([language])
            elif type(language) is list:
                languages.update(language)
        return languages

    def __build_language_tokenizer(self, languages: List[str]) -> Tokenizer:
        if self.language_tokenizer is None:
            self.language_tokenizer = Tokenizer()
            self.language_tokenizer.fit_on_texts(languages)
        return self.language_tokenizer

    def get_files(self, data_dirs: List[str]) -> List[str]:
        result: List[str] = []
        for data_dir in data_dirs:
            depth = 0
            for root, sub_folders, files in os.walk(data_dir):
                depth += 1

                # ignore the first loop
                if depth == 1:
                    continue

                language = os.path.basename(root)
                if self.__should_language_be_loaded(language):
                    result.extend([os.path.join(root, f) for f in files])
                depth += 1
        return result

    def __load_words_from_file(self, file_name: str) -> List[str]:
        try:
            with open(file_name, "r") as f:
                contents = f.read()
        except UnicodeDecodeError:
            logging.warning(f"Encountered UnicodeDecodeError, ignore file {file_name}.")
            return []
        return self.__load_words_from_str(contents)

    def __load_words_from_str(self, s: str) -> List[str]:
        contents = " ".join(s.splitlines())
        result = re.split(r"[{}()\[\]\'\":.*\s,#=_/\\><;?\-|+]", contents)

        # remove empty elements
        result = [word for word in result if word.strip() != ""]

        return result

    def __load_sentence(self, file_name: str) -> str:
        """ Used in loading data, word that is not in the vocabulary will not be included
        """
        return " ".join([word for word in self.__load_words_from_file(file_name) if word in self.vocab])

    def __get_weights(self) -> ndarray:
        vocab_size = len(self.vocab_tokenizer.word_index) + 1
        weight_matrix = zeros((vocab_size, self.wordvec_dimension))
        for word, index in self.vocab_tokenizer.word_index.items():
            weight_matrix[index] = self.word2vec[word]
        return weight_matrix

    def __should_language_be_loaded(self, language):
        for value in ext_lang_dict.values():
            if value == language:
                return True
        return False

    def load_data(self, data_dir: str) -> (ndarray, ndarray):
        files = self.get_files([data_dir])
        x: List[List[int]] = []
        y: List[List[int]] = []
        for f in files:
            language: str = os.path.dirname(f).split(os.path.sep)[-1]
            sentence: str = self.__load_sentence(f)
            x.append(encode_sentence(sentence, self.vocab_tokenizer))
            y.append(get_lang_sequence(language))
        return pad_sequences(x, maxlen=input_length), asarray(y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neural_network_trainer = NeuralNetworkTrainer(
        f"{config.data_dir}/train",
        f"{config.data_dir}/test",
        config.vocab_location,
        config.word2vec_location,
        ext_lang_dict)
    model = neural_network_trainer.build_model()
    # neural_network_trainer.evaluate_model(model)
    neural_network_trainer.evaluate_model(model)
    neural_network_trainer.save_model(model, config.model_file_location, config.weights_file_location)

    evaluate_saved_data("../resources/x_train.txt", "../resources/y_train.txt", model)
