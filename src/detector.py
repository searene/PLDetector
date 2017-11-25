import numpy as np
from keras.preprocessing.sequence import pad_sequences
from src import config
from src.config import input_length
from src.neural_network_trainer import encode_sentence, to_language, \
    load_model, to_binary_list, \
    load_vocab_tokenizer, load_contents, is_in_vocab

vocab_tokenizer = load_vocab_tokenizer(config.vocab_tokenizer_location)


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


def convert_sigmoid_result_to_binary_list(sigmoid_result):
    result = []
    for row in sigmoid_result:
        converted_row = to_binary_list(np.argmax(row), len(row))
        result.append(converted_row)
    return result


if __name__ == "__main__":
    code = """
def test():
    print("something")
"""
    print(detect(code))
    # calculate_accuracy()
