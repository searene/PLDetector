import numpy as np
from keras.preprocessing.sequence import pad_sequences
from src import config
from src.config import input_length
from src.neural_network_trainer import encode_sentence, load_model, to_binary_list, \
    load_vocab_tokenizer, load_contents, is_in_vocab, ext_lang_dict, get_all_languages

vocab_tokenizer = load_vocab_tokenizer(config.vocab_tokenizer_location)


def to_language(binary_list):
    languages = get_all_languages()
    i = np.argmax(binary_list)
    return languages[i]


def get_neural_network_input(code):
    preprocessed_sentence = load_contents(code)
    filtered_sentence = " ".join([word for word in preprocessed_sentence if is_in_vocab(word, vocab_tokenizer)])
    encoded_sentence = encode_sentence(filtered_sentence, vocab_tokenizer)
    return pad_sequences([encoded_sentence], maxlen=input_length)


def detect(code, model=None):
    if model is None:
        model = load_model(config.model_file_location, config.weights_file_location)
    y_proba = model.predict(get_neural_network_input(code))
    return to_language(y_proba)


if __name__ == "__main__":
    code = """
def test():
    print("something")
"""
    print(detect(code))
