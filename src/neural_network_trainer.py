import glob
import logging
import re
from typing import List, Set, Counter, Union, Dict

import os

from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import asarray, ndarray, zeros

from src import config
from src.github_fetcher import ext_lang_dict


class NeuralNetworkTrainer:
    def __init__(self, train_data_dir: str, test_data_dir: str, vocab_location: str, word2vec_location: str, ext_lang_dict: Dict[str, Union[str, List[str]]]):
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.vocab_location = vocab_location
        self.word2vec_location = word2vec_location
        self.vocab_tokenizer: Tokenizer = None
        self.language_tokenizer: Tokenizer = None
        self.wordvec_dimension = 100
        self.ext_lang_dict = ext_lang_dict
        self.lang_seq_dict: ndarray = self.__get_lang_to_seq_dict()

        languages = self.__get_languages(ext_lang_dict)
        self.__build_language_tokenizer(list(languages))

        self.build_and_save_vocabulary()
        self.build_and_save_word2vec_model()

        self.word2vec: Dict[str, ndarray] = self.__load_word2vec()
        self.vocab: Set[str] = self.load_vocab()
        self.vocab_tokenizer: Tokenizer = self.__build_vocab_tokenizer(self.vocab)

    def build_vocabulary(self) -> Counter:
        vocabulary: Counter = Counter()
        files = self.__get_files()
        for f in files:
            words = self.__load_words(f)
            vocabulary.update(words)

        # remove rare words
        min_count = 2
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
        files: List[str] = self.__get_files()
        for f in files:
            words: List[str] = self.__load_words(f)
            all_tokens.append([token for token in words if token in vocabulary])
        logging.info("Building the Word2Vec model...")
        model = Word2Vec(all_tokens, size=100, window=5, workers=8, min_count=1)
        return model

    def build_and_save_word2vec_model(self) -> None:
        vocab = self.load_vocab()
        model = self.build_word2vec_model(vocab)
        logging.info("Saving the word2vec model to the disk...")
        model.wv.save_word2vec_format(self.word2vec_location, binary=False)

    def load_vocab(self) -> Set[str]:
        with open(self.vocab_location) as f:
            words = f.read().splitlines()
        return set(words)

    def build_model(self) -> Sequential:
        weight_matrix: ndarray = self.__get_weights()

        # build the embedding layer
        input_dim = len(self.vocab_tokenizer.word_index) + 1
        output_dim = self.wordvec_dimension
        x_train, y_train = self.__load_data(self.train_data_dir)

        embedding_layer = Embedding(input_dim, output_dim, weights=[weight_matrix], input_length=x_train.shape[1], trainable=False)

        model = Sequential()
        model.add(embedding_layer)
        model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, verbose=2)
        return model

    def evaluate_model(self, model: Sequential) -> None:
        x_test, y_test = self.__load_data(self.test_data_dir)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print('Test Accuracy: %f' % (acc * 100))

    def __get_languages(self, ext_lang_dict: Dict[str, Union[str, List[str]]]) -> Set[str]:
        languages: Set[str] = set()
        for ext, language in ext_lang_dict.items():
            if type(language) is str:
                languages.update([language])
            elif type(language) is list:
                languages.update(language)
        return languages

    def __build_language_tokenizer(self, types: List[str]) -> Tokenizer:
        if self.language_tokenizer is None:
            self.language_tokenizer = Tokenizer()
            self.language_tokenizer.fit_on_texts(types)
        return self.language_tokenizer

    def __get_files(self) -> List[str]:
        result: List[str] = []
        for root, sub_folders, files in os.walk(self.train_data_dir):
            result.extend([os.path.join(root, f) for f in files])
        return result

    def __load_words(self, file_name: str) -> List[str]:
        try:
            with open(file_name, "r") as f:
                contents = f.read()
        except UnicodeDecodeError:
            logging.warning(f"Encountered UnicodeDecodeError, ignore file {file_name}.")
            return []

        contents = " ".join(contents.splitlines())
        result = re.split(r"([{}\(\)\[\]\'\":\.\*\s])", contents)

        # remove empty elements
        result = [word for word in result if word.strip() != ""]

        return result

    def __load_sentence(self, file_name: str) -> str:
        return " ".join(self.__load_words(file_name))

    def __build_vocab_tokenizer(self, vocab: Set[str]) -> Tokenizer:
        if self.vocab_tokenizer is None:
            self.vocab_tokenizer = Tokenizer(lower=False, filters="")
            self.vocab_tokenizer.fit_on_texts(vocab)
        return self.vocab_tokenizer

    def __encode(self, sentences: List[str]) -> ndarray:
        encoded_sentences: List[List[int]] = self.vocab_tokenizer.texts_to_sequences(sentences)
        max_length = max([len(sentence) for sentence in encoded_sentences])
        # shape: n_files * max_length
        return pad_sequences(encoded_sentences, maxlen=max_length, padding="post")

    def __get_weights(self) -> ndarray:
        vocab_size = len(self.vocab_tokenizer.word_index) + 1
        weight_matrix = zeros((vocab_size, self.wordvec_dimension))
        for word, index in self.vocab_tokenizer.word_index.items():
            weight_matrix[index] = self.word2vec[word]
        return weight_matrix

    def __load_word2vec(self) -> Dict[str, ndarray]:
        result: Dict[str, ndarray] = dict()
        with open(self.word2vec_location, "r") as f:
            lines: List[str] = f.readlines()[1:]
        for line in lines:
            parts: List[str] = line.split()
            result[parts[0]] = asarray(parts[1:], dtype="float32")
        return result

    def __get_lang_to_seq_dict(self) -> Dict[str, int]:
        i = 0
        result = {}
        for ext, lang in self.ext_lang_dict.items():
            if type(lang) is list:
                for lang1 in lang:
                    result[lang1] = i
                    i += 1
            else:
                result[lang] = i
                i += 1
        return result

    def __load_data(self, data_dir: str) -> (ndarray, ndarray):
        x_train_raw: List[str] = []
        y_train_raw: List[str] = []
        files: List[str] = glob.glob(f"{data_dir}/**/*")
        for f in files:
            language = os.path.dirname(f).split(os.path.sep)[-1]
            sentence = self.__load_sentence(f)
            x_train_raw.append(sentence)
            y_train_raw.append(language)
        x: ndarray = self.__encode(x_train_raw)
        y: ndarray = asarray([self.lang_seq_dict[lang] for lang in y_train_raw])
        return x, y



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neural_network_trainer = NeuralNetworkTrainer(
        f"{config.data_dir}/train",
        f"{config.data_dir}/test",
        config.vocab_location,
        config.word2vec_location,
        ext_lang_dict)
    model = neural_network_trainer.build_model()
    neural_network_trainer.evaluate_model(model)
