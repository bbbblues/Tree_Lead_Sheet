
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import os
from GT_chord_dict import *
from one_layer import save_path, train_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pitch_input_size = 63
dur_input_size = 20
MAX_LENGTH = 200
beats_per_bar = 4
do_sample = True
train_max_length = 200

model_dir = save_path
gene_path = model_dir + '/gene/'
if not os.path.exists(gene_path):
    os.makedirs(gene_path)

chosen_seed_ol = 125
chosen_model_epoch_ol = '300'
gene_num_ol = 30


class Note_encoder(nn.Module):
    def __init__(self, hidden_size, pitch_input_size, dur_input_size, pitch_embedding_size, dur_embedding_size):
        super(Note_encoder, self).__init__()
        self.hidden_size = hidden_size
        self.pitch_input_size = pitch_input_size
        self.dur_input_size = dur_input_size
        self.input_size = pitch_embedding_size + dur_embedding_size

        self.pitch_embedding = nn.Embedding(pitch_input_size, pitch_embedding_size)
        self.dur_embedding = nn.Embedding(dur_input_size, dur_embedding_size)
        self.lstm = nn.LSTM(self.input_size, hidden_size)

    def forward(self, note, hn, cn):
        embedded_pitch = self.pitch_embedding(note[0]).view(1, 1, -1)
        embedded_dur = self.dur_embedding(note[1]).view(1, 1, -1)
        embedded_notes = torch.cat([embedded_pitch, embedded_dur], -1)
        output, (hn, cn) = self.lstm(embedded_notes, (hn, cn))
        return output, (hn, cn)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Note_decoder(nn.Module):
    def __init__(self, hidden_size, pitch_input_size, dur_input_size, pitch_embedding_size, dur_embedding_size):
        super(Note_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = pitch_embedding_size + dur_embedding_size

        self.pitch_embedding = nn.Embedding(pitch_input_size, pitch_embedding_size)
        self.dur_embedding = nn.Embedding(dur_input_size, dur_embedding_size)
        self.lstm = nn.LSTM(self.input_size, hidden_size)
        self.pitch_out = nn.Linear(hidden_size, pitch_input_size)
        self.dur_out = nn.Linear(pitch_embedding_size+hidden_size, dur_input_size)

    def forward(self, note, hn, cn, true_pitch):
        embedded_pitch = self.pitch_embedding(note[0]).view(1, 1, -1)
        embedded_dur = self.dur_embedding(note[1]).view(1, 1, -1)
        embedded_true_pitch = self.pitch_embedding(true_pitch).view(1, -1)
        input = torch.cat([embedded_pitch, embedded_dur], -1)
        output, (hn, cn) = self.lstm(input, (hn, cn))
        pitch_out = self.pitch_out(output[0])
        dur_out = self.dur_out(torch.cat([embedded_true_pitch, output[0]], -1))
        return (pitch_out, dur_out), (hn, cn)

    def forward_generation(self, note, hn, cn):
        embedded_pitch = self.pitch_embedding(note[0]).view(1, 1, -1)
        embedded_dur = self.dur_embedding(note[1]).view(1, 1, -1)
        input = torch.cat([embedded_pitch, embedded_dur], -1)
        output, (hn, cn) = self.lstm(input, (hn, cn))
        pitch_out = self.pitch_out(output[0])
        pitch_prob = F.softmax(pitch_out, dim=-1)
        generated_pitch_tensor = torch.multinomial(pitch_prob[0], 1)
        generated_pitch = generated_pitch_tensor.item()
        embedded_generated_pitch = self.pitch_embedding(generated_pitch_tensor).view(1, -1)
        dur_out = self.dur_out(torch.cat([embedded_generated_pitch, output[0]], -1))
        dur_prob = F.softmax(dur_out, dim=-1)
        generated_dur = torch.multinomial(dur_prob[0], 1).item()
        return generated_pitch, generated_dur, (hn, cn)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def get_train_set(train_size):
    path = '/Users/bbbblues/Desktop/GT_train_notes_new.pickle'
    notes = pickle.load(open(path, 'rb'))

    notes_set = []
    for song in notes:
        song_notes = []
        for i, item in enumerate(song):
            song_notes.append((int(item[0]-36), beat_dict[item[1]]))
            if i > train_max_length:
                break
        notes_set.append(song_notes)
        #print(i)

    return notes_set[:train_size]


def get_seed(train_set):
    notes_seed = []

    for sample in train_set:
        sample_seed = sample[:10]
        notes_seed.append(sample_seed)

    return notes_seed


def generate(encoder, decoder, max_length=MAX_LENGTH):
    with torch.no_grad():
        decoded_notes = []

        train_set = get_train_set(train_size)
        seeds = get_seed(train_set)
        seed = seeds[0]

        seed_tensor = torch.tensor(seed, dtype=torch.long, device=device)
        seed_length = seed_tensor.size(0)

        enc_hn = encoder.initHidden()
        enc_cn = encoder.initHidden()
        for ei in range(seed_length):
            enc_out, (enc_hn, enc_cn) = encoder(seed_tensor[ei], enc_hn, enc_cn)
        dec_hn = enc_hn
        dec_cn = enc_cn

        note_start_symbol = torch.tensor([pitch_input_size-1, dur_input_size-1], device=device)
        dec_input = note_start_symbol
        for i in range(max_length):
            pitch_ind, dur_ind, (dec_hn, dec_cn) = decoder.forward_generation(dec_input, dec_hn, dec_cn)
            dec_input = torch.tensor([pitch_ind, dur_ind], device=device)

            decoded_pitch = pitch_ind + 36
            decoded_beat = ind_beat_dict[dur_ind]
            decoded_note = [decoded_pitch, decoded_beat]
            decoded_notes.append(decoded_note)

    return decoded_notes


def generate_all():
    test_enc = torch.load(model_dir + 'Enc_Epoch' + chosen_model_epoch_ol + '.pickle')
    test_dec = torch.load(model_dir + 'Dec_Epoch' + chosen_model_epoch_ol + '.pickle')

    for i in range(gene_num_ol):
        notes = generate(test_enc, test_dec)
        pickle.dump(notes, open(gene_path + 'grtd_notes_' + chosen_model_epoch_ol + '_seed' +
                                str(chosen_seed_ol) + '_' + str(i) + '.pickle', 'wb'))


if __name__ == "__main__":
    generate_all()

