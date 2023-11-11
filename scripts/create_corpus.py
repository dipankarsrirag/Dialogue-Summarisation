import pandas as pd
import contractions


def clean(text):
    return contractions.fix(text.lower().replace("\n", "\t"))


train = pd.read_json("../data/raw/dialogsum/dialogsum.train.jsonl", lines=True)
dev = pd.read_json("../data/raw/dialogsum/dialogsum.dev.jsonl", lines=True)

train_corpus = list(train["dialogue"])
dev_corpus = list(dev["dialogue"])

corpus = []
corpus.extend(train_corpus)
corpus.extend(dev_corpus)


corpus = list(map(clean, corpus))
corpus = "\n".join(corpus)

with open("../data/processed/corpus", "w") as f:
    f.write(corpus)
special = [
    "!!",
    "!!!",
    "!'",
    "!)",
    "!.",
    "!?",
    "!]",
    "$,",
    "$.",
    "?!",
    "?'",
    "?)",
    "?),",
    "???",
    "?]",
    "@",
    "[",
    "\\",
    "]",
    "],",
    "].",
    "~",
    ".#",
    ".'",
    ".)",
    ".,",
    "..",
    "...",
    "....",
    "......",
    "...?",
    "...]",
    ".?",
    ".]",
    "/",
    ":",
    "::",
    ";",
    "=",
    ",'",
    ",,",
    ",...",
    "-",
    "-$",
    "-'",
    "--",
    "---",
    "(",
    "($",
    "('",
    "(.",
    ")",
    "),",
    ").",
    "*",
    "**",
    "+",
    "%,",
    "%.",
    "&",
    "'",
    "',",
    "'-",
    "'.",
    "';",
    "'?",
]
