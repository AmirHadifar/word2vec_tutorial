from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(level=logging.INFO)

OUTPUT_FILE_PATH = './'
INPUT_FILE_PATH = './wiki.fa.text'


def train_model():
    sentences = LineSentence(INPUT_FILE_PATH)
    model = Word2Vec(sentences, size=200, window=5, sg=1)
    model.wv.save_word2vec_format(OUTPUT_FILE_PATH + 'word2vec.txt', binary=False)


def load_model():
    wiki_model = KeyedVectors.load_word2vec_format(OUTPUT_FILE_PATH + 'word2vec.txt')
    most_similar = wiki_model.most_similar(u'ایران')
    for words in most_similar:
        print(words[0])


if __name__ == '__main__':
    train_model()
    load_model()
