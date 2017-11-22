# from gensim.models import Word2Vec
# from typing import List
#
# from src.neural_network_trainer import load_word2vec
#
# with open('../resources/vocab_copy.txt') as f:
#     lines = f.read().splitlines()
#
# word = lines[0]
# model = Word2Vec([[word]], size=100, window=1, workers=8, min_count=1)
# model.wv.save_word2vec_format('../resources/word2vec_copy.txt')
#
# word2vec = load_word2vec("../resources/word2vec_copy.txt")
# for key in word2vec:
#     print(key)
#     print(word == key)

with open("../resources/word2vec_copy.txt", "r", encoding="utf-8") as f:
    word = f.readlines()[1].split()[0]
print(word)
