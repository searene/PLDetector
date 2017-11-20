import os

github_username = "searene"
github_password = "password"
proxies = {}
download_location = "/data/code"

current_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(current_dir, "../resources/code")
vocab_location = os.path.join(current_dir, "../resources/vocab.txt")
word2vec_location = os.path.join(current_dir, "../resources/word2vec.txt")
model_file_location = os.path.join(current_dir, "../resources/models/model.json")
weights_file_location = os.path.join(current_dir, "../resources/models/model.h5")

input_length = 500

