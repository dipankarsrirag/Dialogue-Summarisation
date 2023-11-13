import pandas as pd
from nltk.tokenize import WordPunctTokenizer
import json


def combine_columns(row):
    try:
        return row['dialogue'] + ' ' + row['summary']
    except RuntimeError:
        print("PandasDataFrameNotFound")


def fix_contractions(text):
    with open('../data/contractions.json', 'r') as f:
        contractions = json.load(f)
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
    return text


if __name__ == '__main__':
    train = pd.read_json(
        '../data/raw/dialogsum/dialogsum.train.jsonl', lines=True)
    train['summary'] = train['summary'].apply(
        lambda x: '<SOS> ' + x + ' <EOS>')

    corpus = train.apply(combine_columns, axis=1)
    corpus = list(map(tokenize, corpus))
    train_corpus = '\n'.join(corpus)

    with open("../data/processed/train_corpus", "w") as f:
        f.write(train_corpus)
    print('Training Corpus Generated')
    dev = pd.read_json(
        '../data/raw/dialogsum/dialogsum.dev.jsonl', lines=True)
    dev['summary'] = dev['summary'].apply(
        lambda x: '<SOS> ' + x + ' <EOS>')

    dev_corpus = dev.apply(combine_columns, axis=1)
    dev_corpus = list(map(tokenize, dev_corpus))
    dev_corpus = '\n'.join(dev_corpus)
    with open("../data/processed/dev_corpus", "w") as f:
        f.write(dev_corpus)
    corpus.append(dev_corpus)
    print('Dev Corpus Generated')

    test = pd.read_json(
        '../data/raw/dialogsum/dialogsum.test.jsonl', lines=True)[['dialogue', 'summary1']]
    test.columns = ['dialogue', 'summary']
    test['summary'] = test['summary'].apply(
        lambda x: '<SOS> ' + x + ' <EOS>')

    test_corpus = test.apply(combine_columns, axis=1)
    test_corpus = list(map(tokenize, test_corpus))
    test_corpus = '\n'.join(test_corpus)
    corpus.append(test_corpus)
    corpus = '\n'.join(corpus)
    print('Test Corpus Generated')

    with open("../data/processed/glove_corpus", "w") as f:
        f.write(corpus)

    print('GloVe Corpus Generated')