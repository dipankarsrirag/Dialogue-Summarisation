from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def load_glove_model(glove_file_path):
    glove_model = KeyedVectors.load_word2vec_format(
        glove_file_path, binary=False, no_header=True)
    return glove_model


def get_corpus():
    with open("../data/processed/glove_corpus", "r") as f:
        sentences = f.readlines()
    return sentences

if __name__ == '__main__':

    sentences = get_corpus()
    tokens = [sent.split() for sent in sentences]

    base_model = Word2Vec(vector_size=300, min_count=20, epochs=20)
    base_model.build_vocab(tokens)

    total_examples = base_model.corpus_count

    corpus_path = '../embeds/GloVe/glove.corpus.300d.txt'
    corpus_model = load_glove_model(corpus_path)

    base_model.build_vocab(
        [list(corpus_model.key_to_index.keys())], update=True)
    base_model.train(tokens, total_examples=total_examples,
                     epochs=base_model.epochs)

    base_model_wv = base_model.wv
    base_model_wv.save_word2vec_format('../models/GloVe-Word2Vec/glove.bin')
