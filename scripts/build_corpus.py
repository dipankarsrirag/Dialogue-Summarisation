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
    corpus = '\n'.join(corpus)

    with open("../data/processed/corpus", "w") as f:
        f.write(corpus)
