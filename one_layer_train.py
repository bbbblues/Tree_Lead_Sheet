'''
Simple one layer decoder for evaluation.
'''
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hidden_size = 256
pitch_embedding_size = 10
dur_embedding_size = 10
pitch_input_size = 63
dur_input_size = 20
learning_rate = 0.0001
epochs = 2000
train_size = 125
save_epochs = 10
print_loss_epochs = 1
do_shuffle = True
train_set_name = 'GT'

print_every = 60
train_max_length = 120

criterion = nn.CrossEntropyLoss()

save_path = 'note_layer_only_nodel/' + train_set_name + '_Epochs' + str(epochs) + '_TrainSize' + str(train_size) + \
    '_lr' + str(learning_rate) + '_hidden' + str(hidden_size) + '_length' + str(train_max_length) + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m ,s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def savePlot(points, plot_name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.5)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(save_path + plot_name)


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
        self.dur_out = nn.Linear(pitch_embedding_size + hidden_size, dur_input_size)

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
        pass

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(seed_tensor, train_tensor, encoder, decoder, optimizer, criterion):
    enc_hn = encoder.initHidden()
    enc_cn = encoder.initHidden()

    optimizer.zero_grad()

    seed_length = seed_tensor.size(0)
    train_length = train_tensor.size(0) 

    encoder_outputs = torch.zeros(seed_length, encoder.hidden_size, device=device)
    for ei in range(seed_length):
        enc_out, (enc_hn, enc_cn) = encoder(seed_tensor[ei], enc_hn, enc_cn)
        encoder_outputs[ei] = enc_out[0, 0]
    dec_hn = enc_hn
    dec_cn = enc_cn

    note_start_symbol = torch.tensor([pitch_input_size-1, dur_input_size-1], device=device)
    dec_input = note_start_symbol

    loss = 0
    save_loss = 0
    for i in range(train_length):
        (pitch_out, dur_out), (dec_hn, dec_cn) = decoder(dec_input, dec_hn, dec_cn, train_tensor[i][0])
        # print(train_tensor[i][0])
        # print(pitch_out)
        loss += criterion(pitch_out, torch.tensor([train_tensor[i][0].item()]))
        loss += criterion(dur_out, torch.tensor([train_tensor[i][1].item()]))

        if (i+1) % 50 == 0:
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            save_loss += loss 
            loss = 0

        dec_input = train_tensor[i]

    if loss != 0:
        loss.backward(retain_graph=True)
        optimizer.step()
        save_loss += loss

    save_loss /= train_length

    return save_loss


def trainIters(note_encoder, note_decoder, training_set, optimizer, print_every, plot_losses):
    start = time.time()
    print_loss = 0
    plot_loss = 0

    if do_shuffle:
        np.random.shuffle(training_set)

    # seeds need to be the same order
    seeds = get_seed(training_set)

    for iter in range(1, train_size+1):
        train_sample = training_set[iter-1]
        seed = seeds[iter-1]
        train_tensor = torch.tensor(train_sample, dtype=torch.long, device=device)
        seed_tensor = torch.tensor(seed, dtype=torch.long, device=device)

        loss = train(seed_tensor, train_tensor, note_encoder, note_decoder, optimizer, criterion)

        print_loss += loss
        plot_loss += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss / print_every
            print_loss = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / train_size),
                iter, iter/train_size*100, print_loss_avg))

    plot_loss_avg = plot_loss / train_size
    plot_losses.append(plot_loss_avg.item())

    return plot_losses


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
        # print(i)

    return notes_set[:train_size]


def get_seed(train_set):
    notes_seed = []

    for sample in train_set:
        sample_seed = sample[:10]
        notes_seed.append(sample_seed)

    return notes_seed


def start():
    encoder = Note_encoder(hidden_size, pitch_input_size, dur_input_size, pitch_embedding_size, dur_embedding_size)
    decoder = Note_decoder(hidden_size, pitch_input_size, dur_input_size, pitch_embedding_size, dur_embedding_size)
    all_parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(all_parameters, lr=learning_rate, betas=(0.99, 0.999))

    train_set = get_train_set(train_size)
    plot_losses = []

    for i in range(epochs):
        plot_losses = trainIters(encoder, decoder, train_set, optimizer, print_every, plot_losses)
        savePlot(plot_losses, 'loss.png')
        pickle.dump(plot_losses, open(save_path + 'losses.pickle', 'wb'))
        enc_save_path = save_path + 'Enc_Epoch' + str(i+1) + '.pickle'
        dec_save_path = save_path + 'Dec_Epoch' + str(i+1) + '.pickle'

        if ((i+1) % save_epochs == 0 and plot_losses[-1] < 0.2) or (i+1) % 100 == 0:
            torch.save(encoder, enc_save_path)
            torch.save(decoder, dec_save_path)

        if (i+1) % print_loss_epochs == 0:
            print("Epochs: " + str(i+1))
            print('loss:' + str(plot_losses[-1]))
            print('\n')

if __name__ == "__main__":
    start()





        



    







