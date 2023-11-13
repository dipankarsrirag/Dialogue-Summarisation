from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from gensim.models import KeyedVectors
import json


def load_glove_model(glove_file_path):
    glove_model = KeyedVectors.load_word2vec_format(
        glove_file_path, binary=False, no_header=True)
    return glove_model


def combine_columns(row):
    return row['dialogue'] + ' ' + row['summary']


def fix_contractions(text):
    tokens = text.split()
    cleaned = []
    for token in tokens:
        cleaned.append(contractions.get(token, token))
    return ' '.join(cleaned)


def tokenize(text):
    tokenizer = WordPunctTokenizer()
    text = fix_contractions(text)
    tokens = tokenizer.tokenize(text)
    text = ' '.join(tokens).lower()
    text = text.replace('# person1 #', '#person1#')
    text = text.replace('# person2 #', '#person2#')
    text = text.replace('# person3 #', '#person3#')
    text = text.replace('# person4 #', '#person4#')
    text = text.replace('# person5 #', '#person5#')
    text = text.replace('# person6 #', '#person6#')
    text = text.replace('# person7 #', '#person7#')
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' ?', '?')
    text = text.replace(' !', '!')
    text = text.replace(" ' ", "'")
    text = text.replace("< ", "<")
    text = text.replace(" >", ">")
    return text.split()


if __name__ == '__main__':

    train = pd.read_json('../data/raw/dialogsum/dialogsum.train.jsonl',
                         lines=True)[['dialogue', 'summary']]

    with open('../data/contractions.json', 'r') as f:
        contractions = json.load(f)

    train['summary'] = train['summary'].apply(
        lambda x: '<SOS> ' + x + ' <EOS>')
    concat = train.apply(combine_columns, axis=1)

    tokens = list(concat.apply(lambda x: tokenize(x)))

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
