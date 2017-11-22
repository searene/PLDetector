from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from numpy import ndarray
from numpy import asarray
from keras import Sequential
from typing import List

from src.neural_network_trainer import to_language, get_lang_sequence

weight_matrix = asarray([
    [1, 10],
    [2, 10],
    [3, 2],
    [3, 7],
    [3, 10],
    [10, 2],
    [5, 2]
])
x_train: ndarray = asarray([
    [0, 1, 2, 5, 2, 1],
    [1, 2, 1, 4, 2, 3],
    [1, 3, 6, 4, 2, 2],
    [1, 1, 1, 4, 2, 4],
    [1, 2, 5, 4, 2, 5],
    [1, 3, 6, 4, 2, 2],
    [1, 1, 1, 4, 2, 4],
    [1, 2, 5, 4, 2, 5],
    [1, 3, 6, 4, 2, 2],
    [1, 1, 1, 4, 2, 4],
    [1, 2, 5, 4, 2, 5],
])
y_train: ndarray = asarray([
    [0, 1],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])


def get_prediction_value_from_sigmoid_result(sigmoid_result: ndarray) -> ndarray:
    sigmoid_result_list: List[List[float]] = ndarray.tolist(sigmoid_result)
    prediction_value_list: List[List[int]] = []
    for row in sigmoid_result_list:
        if row[0] < row[1]:
            prediction_value_list.append([0, 1])
        else:
            prediction_value_list.append([1, 0])
    return asarray(prediction_value_list)


def build_and_evaluate_mode() -> Sequential:
    model = Sequential()
    embedding_layer = Embedding(7, 2, weights=[weight_matrix], input_length=6,
                                trainable=False)
    model.add(embedding_layer)
    model.add(Conv1D(filters=128, kernel_size=2, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(2, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, verbose=2)
    loss, acc = model.evaluate(x_train, y_train, verbose=2)
    print(f"loss: {loss}, acc: {acc}")
    return model


def evaluate_model_manually(model: Sequential) -> None:
    total_count = 0
    correct_count = 0
    predicted_y: ndarray = model.predict(x_train)
    converted_predicted_y: ndarray = get_prediction_value_from_sigmoid_result(predicted_y)
    for correct_value, predicted_value in zip(ndarray.tolist(y_train), ndarray.tolist(converted_predicted_y)):
        if correct_value == predicted_value:
            correct_count += 1
        total_count += 1
    print(f"accuracy calculated manually: {correct_count / total_count}")


if __name__ == "__main__":
    # model: Sequential = build_and_evaluate_mode()
    # evaluate_model_manually(model)

    languages = ['C', 'C#', 'CSS', 'HTML', 'Java', 'Javascript', 'Python']
    for lang in languages:
        if to_language(get_lang_sequence(lang)) != lang:
            raise Exception("Not equal.")
