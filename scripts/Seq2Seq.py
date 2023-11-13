from __future__ import unicode_literals, print_function, division
from gensim.models import KeyedVectors
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
from google.colab import drive

from io import open

import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')

MAX_LENGTH = 300

drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_matrix=None, bidirectional=True, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.bidirectional = bidirectional

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=False, padding_idx=PAD_IDX)
        else:
            self.embedding = nn.Embedding(
                input_size, hidden_size, padding_idx=PAD_IDX)

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)

        if self.bidirectional:
            batch, seq_len, _ = output.shape
            output = output.view(
                (batch, seq_len, self.num_directions, self.hidden_size))

            hidden = hidden.view(
                self.num_layers, self.num_directions, batch, self.hidden_size)
            output = output[:, :, 0, :]+output[:, :, 1, :]
            hidden = hidden[:, 0, :, :] + hidden[0, 1, :, :]

        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, embedding_matrix=None, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=False, padding_idx=PAD_IDX)
        else:
            self.embedding = nn.Embedding(
                output_size, hidden_size, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size,
                          num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None and i < target_tensor.size(1):
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        last_layer_hidden = hidden[-1].unsqueeze(0)
        query = last_layer_hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


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
    return text.split()


def prepareData(src, trg):
    dial = np.array(src)
    summary = np.array(trg)
    pairs = [[dial[i], summary[i]] for i in range(len(dial))]
    return pairs


def indexesFromSentence(sentence):
    return [dictionary.get(word) for word in tokenize(sentence) if word in dictionary.keys()]


def tensorFromSentence(sentence):
    indexes = indexesFromSentence(sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0])
    target_tensor = tensorFromSentence(pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(pairs, batch_size):

    num_pairs = len(pairs)

    input_ids = np.full((num_pairs, MAX_LENGTH), PAD_IDX, dtype=np.int32)
    target_ids = np.full((num_pairs, MAX_LENGTH), PAD_IDX, dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(inp)
        tgt_ids = indexesFromSentence(tgt)
        inp_ids.append(EOS_TOKEN)
        tgt_ids.append(EOS_TOKEN)
        input_ids[idx, :len(inp_ids)] = inp_ids[:MAX_LENGTH]
        target_ids[idx, :len(tgt_ids)] = tgt_ids[:MAX_LENGTH]

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(
            encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
          print_every=100, plot_every=100, momentum=0.9):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    # Using Adam with momentum
    encoder_optimizer = optim.Adam(
        encoder.parameters(), lr=learning_rate)  # , momentum=momentum)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate)  # , momentum=momentum)
    scheduler_encoder = optim.lr_scheduler.StepLR(
        encoder_optimizer, step_size=10, gamma=0.5)
    scheduler_decoder = optim.lr_scheduler.StepLR(
        decoder_optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder,
                           encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        scheduler_encoder.step()
        scheduler_decoder.step()

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                f'{timeSince(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100:.2f}%) {print_loss_avg:.4f}')

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            # Checkpointing
            if plot_loss_avg < best_loss:
                best_loss = plot_loss_avg
                torch.save(encoder.state_dict(), 'encoder_best.pth')
                torch.save(decoder.state_dict(), 'decoder_best.pth')


def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence(sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                decoded_words.append('<eos>')
                break
            decoded_words.append(vocab[idx.item()])
    return decoded_words, decoder_attn


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    df = pd.read_json('/content/drive/My Drive/COMP9444/data/raw/dialogsum/dialogsum.train.jsonl',
                      lines=True)[['dialogue', 'summary']]
    src_tokens = list(df['dialogue'].apply(lambda x: tokenize(x)))
    trg_tokens = list(df['summary'].apply(lambda x: tokenize(x)))

    src = [' '.join(sent) for sent in src_tokens]
    trg = [' '.join(sent) for sent in trg_tokens]

    word_vectors = KeyedVectors.load_word2vec_format(
        '/content/drive/My Drive/COMP9444/models/GloVe-Word2Vec/glove.bin')

    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"
    if pad_token not in word_vectors.key_to_index:
        pad_index = len(word_vectors)
        word_vectors.key_to_index[pad_token] = pad_index
        word_vectors.index_to_key.append(pad_token)
    else:
        pad_index = word_vectors.key_to_index[pad_token]

    PAD_IDX = pad_index
    SOS_TOKEN = word_vectors.key_to_index['<sos>']
    EOS_TOKEN = word_vectors.key_to_index['<eos>']

    vocab = list(word_vectors.key_to_index.keys())
    vocab_size = len(vocab)
    embedding_dim = word_vectors.get_vector('<sos>').shape[0]
    dictionary = word_vectors.key_to_index

    embedding_matrix = torch.zeros(vocab_size, embedding_dim)

    for i, word in enumerate(vocab):
        if word == pad_token:
            continue
        embedding_matrix[i] = torch.Tensor(np.array(word_vectors[word]))

    embedding_matrix = embedding_matrix.to(device)

    hidden_size = 300
    batch_size = 32
    num_layers = 3

    pairs = prepareData(src, trg)
    train_dataloader = get_dataloader(pairs, batch_size)

    encoder = EncoderRNN(vocab_size, hidden_size, num_layers,
                         embedding_matrix).to(device)
    decoder = AttnDecoderRNN(hidden_size, vocab_size,
                             num_layers, embedding_matrix).to(device)

    total_params_encoder = count_trainable_parameters(encoder)
    total_params_decoder = count_trainable_parameters(decoder)
    total_params_attention = count_trainable_parameters(decoder.attention)

    total_trainable_params = total_params_encoder + \
        total_params_decoder+total_params_attention
    train(train_dataloader, encoder, decoder, 30, print_every=1, plot_every=1)
