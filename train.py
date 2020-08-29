
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
chord_embedding_size = 10
pitch_embedding_size = 10
dur_embedding_size = 10
chord_input_size = 158
pitch_input_size = 63
dur_input_size = 16
learning_rate = 0.0001
epochs = 1000
train_size = 125
print_every_iters = train_size
plot_every_iters = train_size
# teacher_forcing_ratio = 0.5
train_set_name = 'GTN'
save_epochs = 10
print_loss_epochs = 1
do_shuffle = True
chord_as_seed = True

CHDC_MAX_NUM = 3
# CHI_MAX_LENGTH = 30
bar_line = '|'
beats_per_bar = 4

# train_chord_path = 'train_chords.pickle'
save_path = 'chord_model/' + 'sec_feed_note_' + train_set_name + '_Epochs' + str(epochs) + '_TrainSize' + \
            str(train_size) + '_lr' + str(learning_rate) + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


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


def savePlot(points, plot_name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.5)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(save_path + plot_name)


# single_layer LSTM encoder
class Chord_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(Chord_encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)

    def forward(self, input, hn, cn):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, (hn, cn) = self.lstm(output, (hn, cn))
        return output, (hn, cn)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# section layer and chord layer decoder 
class Chord_decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size):
        super(Chord_decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn_redu = nn.Linear(hidden_size, embedding_size)
        # self.log_softmax = nn.LogSoftmax()
        # self.merge = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, input, hn, cn, encoder_outputs):
        output = self.embedding(input).view(1, 1, -1)
        # output = F.relu(output)

        # attention layer
        attn_weights = torch.zeros(len(encoder_outputs))
        for i in range(len(encoder_outputs)):
            weight = hn.squeeze().dot(encoder_outputs[i].squeeze())
            attn_weights[i] = weight
        attn_weights = (attn_weights.unsqueeze(0)).unsqueeze(0)  # 1 * 1 * enc_length
        attn_weights = nn.Softmax()(attn_weights)
        encoder_outputs = encoder_outputs.unsqueeze(0)
        context = attn_weights.bmm(encoder_outputs)  # 1 * 1 * hidden_size
        context = self.attn_redu(context)
        output = torch.cat([context, output], -1)

        output, (hn, cn) = self.lstm(output, (hn, cn))
        # output = self.log_softmax(self.out(output[0]))
        output = self.out(output[0])
        return output, (hn, cn)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# note layer decoder 
class Note_decoder(nn.Module):
    def __init__(self, hidden_size, chord_input_size, pitch_input_size, dur_input_size,
                 chord_embedding_size, pitch_embedding_size, dur_embedding_size):
        super(Note_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.note_embedding_size = pitch_embedding_size + dur_embedding_size
        self.context_size = chord_embedding_size
        self.sec_size = chord_embedding_size
        self.input_size = chord_embedding_size + self.note_embedding_size + self.context_size + self.sec_size

        self.chord_embedding = nn.Embedding(chord_input_size, chord_embedding_size)
        self.pitch_embedding = nn.Embedding(pitch_input_size, pitch_embedding_size)
        self.dur_embedding = nn.Embedding(dur_input_size, dur_embedding_size)
        self.lstm = nn.LSTM(self.input_size, hidden_size)
        self.pitch_out = nn.Linear(hidden_size, pitch_input_size)
        self.dur_out = nn.Linear(pitch_embedding_size + hidden_size, dur_input_size)

        self.attn_redu = nn.Linear(hidden_size, chord_embedding_size)

    '''
    def forward_generation(self, chord, note, hn, cn):
        embedded_chord = F.relu(self.chord_embedding(chord).view(1, 1, -1))
        embedded_pitch = F.relu(self.pitch_embedding(note[0]).view(1, 1, -1))
        embedded_dur = F.relu(self.dur_embedding(note[1]).view(1, 1, -1))
        input = torch.cat([embedded_pitch, embedded_dur, embedded_chord], -1)
        output, (hn, cn) = self.lstm(input, (hn, cn))
        pitch_out = self.pitch_out(output[0])
        topv, topi = F.softmax(pitch_out, dim=-1).data.topk(1) ### if sampling, this need to change
        generated_pitch = topi.squeeze().detach()
        embedded_generated_pitch = self.pitch_embedding(generated_pitch).view(1, -1)
        dur_out = self.dur_out(torch.cat([embedded_generated_pitch, output[0]], -1))
        return (pitch_out, dur_out), (hn, cn)
    '''

    def forward(self, sec, chord, note, hn, cn, true_pitch, encoder_outputs):
        embedded_chord = self.chord_embedding(chord).view(1, 1, -1)
        embedded_pitch = self.pitch_embedding(note[0]).view(1, 1, -1)
        embedded_dur = self.dur_embedding(note[1]).view(1, 1, -1)
        embedded_true_pitch = self.pitch_embedding(true_pitch).view(1, -1)
        embedded_sec = self.chord_embedding(sec).view(1, 1, -1)

        # attention_layer
        attn_weights = torch.zeros(len(encoder_outputs))
        for i in range(len(encoder_outputs)):
            weight = hn.squeeze().dot(encoder_outputs[i].squeeze())
            attn_weights[i] = weight
        attn_weights = (attn_weights.unsqueeze(0)).unsqueeze(0)  # 1 * 1 * enc_length
        attn_weights = nn.Softmax()(attn_weights)
        encoder_outputs = encoder_outputs.unsqueeze(0)
        context = attn_weights.bmm(encoder_outputs)  # 1 * 1 * hidden_size
        context = self.attn_redu(context)

        input = torch.cat([embedded_pitch, embedded_dur, embedded_chord, embedded_sec, context], -1)
        output, (hn, cn) = self.lstm(input, (hn, cn))
        pitch_out = self.pitch_out(output[0])
        dur_out = self.dur_out(torch.cat([embedded_true_pitch, output[0]], -1))
        # pitch_out = self.softmax(pitch_out)
        # dur_out = self.softmax(dur_out)
        return (pitch_out, dur_out), (hn, cn)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# train function for a single epoch
def train(seed_tensor, train_tensor, note_train_tensor, note_train_sample, encoder,
          section_decoder, chord_decoders, note_decoder, upper_optimizer, note_optimizer, criterion):
    encoder_hn = encoder.initHidden()
    encoder_cn = encoder.initHidden()

    # encoder_optimizer.zero_grad()
    # section_optimizer.zero_grad()
    # for i in range(SEC_MAX_LENGTH):
    #     chord_optimizers[i].zero_grad()
    upper_optimizer.zero_grad()
    note_optimizer.zero_grad()

    seed_length = seed_tensor.size(0)
    train_length = train_tensor.size(0)
    note_train_length = note_train_tensor.size(0)

    encoder_outputs = torch.zeros(seed_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(seed_length):
        encoder_output, (encoder_hn, encoder_cn) = encoder(
            seed_tensor[ei], encoder_hn, encoder_cn)
        encoder_outputs[ei] = encoder_output[0, 0]

    section_input = torch.tensor([[chords_index_dict['<SOC>']]], device=device)

    section_hn = encoder_hn
    section_cn = encoder_cn

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = True

    child_accum = 0
    ind = 1
    section_ind = 0
    while ind != train_length:
        section_output, (section_hn, section_cn) = section_decoder(section_input, section_hn, section_cn,
                                                                   encoder_outputs)
        loss += criterion(section_output, train_tensor[ind])

        if train_tensor[ind - 1] >= chords_index_dict['A:']:
            child_accum = 0
            if train_tensor[ind - 1] >= chords_index_dict['A:'] and train_tensor[ind - 1] <= chords_index_dict['A\'\'\':']:
                chord_lstm = chord_decoders[0]
            elif train_tensor[ind - 1] >= chords_index_dict['B:'] and train_tensor[ind - 1] <= chords_index_dict['B\'\':']:
                chord_lstm = chord_decoders[1]
            else:
                chord_lstm = chord_decoders[2]
            loss, child_accum, child_hn, child_cn = child_train(
                chord_lstm=chord_lstm, chord_train_tensor=train_tensor[ind + 1:],
                section_hn=section_hn, section_cn=section_cn, ini_input=train_tensor[ind],
                loss=loss, criterion=criterion, use_teacher_forcing=use_teacher_forcing,
                child_accum=child_accum, encoder_outputs=encoder_outputs)
            section_input = torch.tensor([[chords_index_dict['<EOS>']]], device=device)

            # section_hn = torch.cat([section_hn, child_hn], -1)
            # section_cn = torch.cat([section_cn, child_cn], -1)

            ind += child_accum + 1
            section_ind += 1
        else:
            section_input = train_tensor[ind]
            # print(section_input, ind)
            ind += 1

        # if use_teacher_forcing:
        # section_input = train_tensor[ind+child_accum+1]
        # else:
        #    topv, topi = section_output.topk(1)
        #    section_input = topi.squeeze().detach()
        # section_hidden = torch.cat((section_hidden, child_hidden), )

    note_start_symbol = torch.tensor([pitch_input_size - 1, dur_input_size - 1], device=device)
    loss, note_loss, real_note_length = grandchild_train(lstm=note_decoder, chord_train_tensor=train_tensor,
                                                         note_train_tensor=note_train_tensor,
                                                         note_train_sample=note_train_sample, ini_hn=encoder_hn,
                                                         ini_cn=encoder_cn, ini_input=note_start_symbol, loss=loss,
                                                         criterion=criterion, use_teacher_forcing=use_teacher_forcing,
                                                         note_optimizer=note_optimizer,
                                                         encoder_outputs=encoder_outputs)

    loss.backward()

    # encoder_optimizer.step()
    # section_optimizer.step()
    # for i in range(section_ind):
    #    chord_optimizers[i].step()
    # note_optimizer.step()
    upper_optimizer.step()
    upper_optimizer.zero_grad()

    return (loss.item() + note_loss * real_note_length) / (train_length + real_note_length), note_loss


# chord layer train function
def child_train(chord_lstm, chord_train_tensor, section_hn, section_cn,
                ini_input, loss, criterion, use_teacher_forcing, child_accum, encoder_outputs):
    hn = section_hn
    cn = section_cn
    input_tensor = ini_input

    for i in range(chord_train_tensor.size(0)):
        # print(chord_train_tensor[i])
        output, (hn, cn) = chord_lstm(input_tensor, hn, cn, encoder_outputs)
        loss += criterion(output, chord_train_tensor[i])
        child_accum += 1
        if chord_train_tensor[i] == chords_index_dict['<EOS>']:
            break

        if use_teacher_forcing:
            input_tensor = chord_train_tensor[i]
        else:
            topv, topi = output.topk(1)
            input_tensor = topi.squeeze().detach()
        

    return loss, child_accum, hn, cn


# note layer train function
def grandchild_train(lstm, chord_train_tensor, note_train_tensor, note_train_sample,
                     ini_hn, ini_cn, ini_input, loss, criterion, use_teacher_forcing, note_optimizer, encoder_outputs):
    input_tensor = ini_input
    hn = ini_hn
    cn = ini_cn
    extra_beats = 0
    note_ind = 0

    note_loss = 0
    save_note_loss = 0
    back_times = 0

    for i in range(chord_train_tensor.size(0)):
        cur_chord = chord_train_tensor[i]
        if i != (chord_train_tensor.size(0) - 1):
            nxt_chord = chord_train_tensor[i + 1]
        if i != 0:
            lst_chord = chord_train_tensor[i - 1]

        if cur_chord > chords_index_dict['|']:
            cur_sec = cur_chord
        elif cur_chord < chords_index_dict['<SOC>']:
            if nxt_chord == chords_index_dict[bar_line] or nxt_chord == chords_index_dict['<EOS>']:
                if lst_chord == chords_index_dict[bar_line] or lst_chord == chords_index_dict['<SOS>']:
                    total_beats = beats_per_bar
                else:
                    total_beats = beats_per_bar / 2
            else:
                total_beats = beats_per_bar / 2

            beat = extra_beats

            while (note_ind != note_train_tensor.size(0)):
                if beat >= total_beats:
                    extra_beats = beat - total_beats
                    break

                (pitch_out, dur_out), (hn, cn) = lstm(cur_sec, cur_chord, input_tensor, hn, cn, note_train_tensor[note_ind][0],
                                                      encoder_outputs)
                # loss += criterion(pitch_out, torch.tensor([note_train_sample[note_ind][0]], device=device))
                # loss += criterion(dur_out, torch.tensor([note_train_sample[note_ind][1]], device=device))
                beat += ind_beat_dict[note_train_sample[note_ind][1]]

                note_loss += criterion(pitch_out, torch.tensor([note_train_sample[note_ind][0]], device=device))
                note_loss += criterion(dur_out, torch.tensor([note_train_sample[note_ind][1]], device=device))

                if use_teacher_forcing:
                    input_tensor = note_train_tensor[note_ind]
                else:
                    topv_pitch, topi_pitch = pitch_out.topk(1)
                    pitch_ind = topi_pitch.item()
                    topv_dur, topi_dur = dur_out.topk(1)
                    dur_ind = topi_dur.item()
                    input_tensor = torch.tensor([pitch_ind, dur_ind], device=device)

                    '''
                    input_tensor = torch.zeros(pitch_input_size + dur_input_size)
                    input_tensor[pitch_ind] = 1
                    input_tensor[pitch_input_size + dur_ind] = 1
                    '''

                if (note_ind + 1) % 50 == 0:
                    note_loss.backward(retain_graph=True)
                    note_optimizer.step()
                    note_optimizer.zero_grad()
                    save_note_loss += note_loss / 50
                    back_times += 1
                    note_loss = 0

                note_ind += 1

                if beat >= total_beats:
                    extra_beats = beat - total_beats
                    # print(note_ind)
                    break

        if note_ind == note_train_tensor.size(0):
            break

    if note_loss != 0:
        note_loss.backward(retain_graph=True)
        note_optimizer.step()
        if note_ind % 50 == 0:
            save_note_loss += note_loss / 50
        else:
            save_note_loss += note_loss / (note_ind % 50)
        back_times += 1
    # print(back_times, save_note_loss.item() / back_times)
    return loss, save_note_loss.item() / back_times, note_ind


# train for many epochs
def trainIters(encoder, section_decoder, chord_decoders, note_decoder, train_size,
               upper_optimizer, note_optimizer, plot_losses, print_every=print_every_iters,
               plot_every=plot_every_iters):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    chord_train_set = get_chord_train_set(train_size)
    note_train_set = get_note_train_set(train_size)

    if do_shuffle:
        ordered_index = np.arange(train_size)
        np.random.shuffle(ordered_index)
        shuffled_chord = []
        shuffled_note = []
        for i in range(train_size):
            shuffled_chord.append(chord_train_set[ordered_index[i]])
            shuffled_note.append(note_train_set[ordered_index[i]])
        chord_train_set = shuffled_chord
        note_train_set = shuffled_note

    seeds = get_seed(chord_train_set)
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, train_size + 1):
        # chord_train_sample = chord_train_set[iter-1][:27]
        # note_train_sample = note_train_set[iter-1][5:37]
        chord_train_sample = chord_train_set[iter - 1]
        note_train_sample = note_train_set[iter - 1]
        seed = seeds[iter - 1]
        chord_train_tensor = torch.tensor(chord_train_sample, dtype=torch.long, device=device).view(-1, 1)
        note_train_tensor = torch.tensor(note_train_sample, dtype=torch.long, device=device)
        # print(note_train_tensor)

        '''
        note_train_tensor = torch.zeros(len(note_train_sample), pitch_input_size+dur_input_size)
        for i, note in enumerate(note_train_tensor):
            pitch_ind = note_train_sample[i][0]
            dur_ind = note_train_sample[i][1] + pitch_input_size
            note[pitch_ind] = 1
            note[dur_ind] = 1
        '''

        seed_tensor = torch.tensor(seed, dtype=torch.long, device=device).view(-1, 1)

        loss, note_only_loss = train(seed_tensor, chord_train_tensor, note_train_tensor, note_train_sample, encoder,
                                     section_decoder, chord_decoders, note_decoder, upper_optimizer, note_optimizer,
                                     criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / train_size),
                                         iter, iter / train_size * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses, note_only_loss


def get_chord_train_set(train_size):
    '''
    path = '/Users/bbbblues/Desktop/test_chords_index/Eleanor Rigby.10.mid.pickle'
    chords_index = pickle.load(open(path, 'rb'))
    chords = [index_chords_dict[index] for index in chords_index]
    return test_index
    '''

    # path = '/home/peizhe/train_chords.pickle'
    path = '/Users/bbbblues/Desktop/GT_train_chords_new.pickle'
    chord_set = pickle.load(open(path, 'rb'))
    return chord_set[:train_size]


def get_note_train_set(train_size):
    # notes_path = '/home/peizhe/train_notes.pickle'
    notes_path = '/Users/bbbblues/Desktop/GT_train_notes_new.pickle'
    notes = pickle.load(open(notes_path, 'rb'))

    notes_set = []
    for i, song in enumerate(notes):
        song_notes = []
        for item in song:
            song_notes.append((int(item[0] - 36), beat_dict[item[1]]))
        notes_set.append(song_notes)

    return notes_set[:train_size]


def get_seed(train_chords):
    if chord_as_seed:
        seed_chords = []
        for sample in train_chords:
            sample_seeds = []
            for i, chord in enumerate(sample):
                if chord >= chords_index_dict['A:']:
                    for j in range(i, i + 6):
                        if j < len(sample) and sample[j] != chords_index_dict['|']:
                            sample_seeds.append(sample[j])
            seed_chords.append(sample_seeds)
    '''
    else:
        # note as seed
        seed_notes = []
        for sample in train_notes:
            sample_notes = [] 
            for i, note in enumerate(sample):
    '''
    return seed_chords


def start():
    encoder = Chord_encoder(chord_input_size, hidden_size, chord_embedding_size).to(device)
    section_decoder = Chord_decoder(hidden_size, chord_input_size, chord_embedding_size).to(device)
    chord_decoders = [Chord_decoder(hidden_size, chord_input_size,
                                    chord_embedding_size).to(device) for i in range(CHDC_MAX_NUM)]
    note_decoder = Note_decoder(hidden_size, chord_input_size, pitch_input_size,
                                dur_input_size, chord_embedding_size, pitch_embedding_size, dur_embedding_size).to(
        device)
    # encoder = torch.load(save_path + 'Enc_Epoch400.pickle')
    # section_decoder = torch.load(save_path + 'Sec_Epoch400.pickle')
    # chord_decoders = [torch.load(save_path + 'Chord' + str(i) + '_Epoch400.pickle') for i in range(CHDC_MAX_NUM)]
    # note_decoder = torch.load(save_path + 'Note_Epoch400.pickle')

    # encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.99, 0.999))
    # section_optimizer = optim.Adam(section_decoder.parameters(), lr=learning_rate, betas=(0.99, 0.999))
    # chord_optimizers = [optim.Adam(chord_decoders[i].parameters(),
    #     lr=learning_rate, betas=(0.99, 0.999)) for i in range(len(chord_decoders))]
    note_optimizer = optim.Adam(note_decoder.parameters(), lr=learning_rate, betas=(0.99, 0.999))
    all_parameters = list(encoder.parameters()) + list(section_decoder.parameters())

    for i in range(CHDC_MAX_NUM):
        all_parameters += list(chord_decoders[i].parameters())

    # all_parameters = torch.cat([encoder.parameters(), section_decoder.parameters()], -1)
    # for i in range(SEC_MAX_LENGTH):
    #     all_parameters = torch.cat([all_parameters, chord_decoders[i].parameters()], -1)
    upper_optimizer = optim.Adam(all_parameters, lr=learning_rate, betas=(0.99, 0.999))

    # plot_losses = pickle.load(open(save_path + 'total_losses.pickle', 'rb'))
    # note_losses = pickle.load(open(save_path + 'note_losses.pickle', 'rb'))
    plot_losses = []
    note_losses = []
    for i in range(epochs):
        plot_losses, note_only_loss = trainIters(encoder, section_decoder, chord_decoders, note_decoder, train_size,
                                                 upper_optimizer, note_optimizer, plot_losses)
        note_losses.append(note_only_loss)
        savePlot(plot_losses, 'total_loss.png')
        savePlot(note_losses, 'note_loss.png')
        pickle.dump(plot_losses, open(save_path + 'total_losses.pickle', 'wb'))
        pickle.dump(note_losses, open(save_path + 'note_losses.pickle', 'wb'))

        enc_save_path = save_path + 'Enc_Epoch' + str(i + 1) + '.pickle'
        sec_save_path = save_path + 'Sec_Epoch' + str(i + 1) + '.pickle'
        chord_save_paths = [(save_path + 'Chord' + str(j) + '_Epoch' +
                             str(i + 1) + '.pickle') for j in range(CHDC_MAX_NUM)]
        note_save_path = save_path + 'Note_Epoch' + str(i + 1) + '.pickle'

        if ((i + 1) % save_epochs == 0 and plot_losses[-1] < 0.01) or (i + 1) % 100 == 0:
            torch.save(encoder, enc_save_path)
            torch.save(section_decoder, sec_save_path)
            for k in range(CHDC_MAX_NUM):
                torch.save(chord_decoders[k], chord_save_paths[k])
            torch.save(note_decoder, note_save_path)

            # chords, notes = evaluate(encoder, section_decoder, chord_decoders, note_decoder)
            # pickle.dump(chords, open(save_path + 'chords_' + str(i+1) + '.pickle', 'wb'))
            # pickle.dump(notes, open(save_path + 'notes_' + str(i+1) + '.pickle', 'wb'))

        if (i + 1) % print_loss_epochs == 0:
            print("Epochs: " + str(i + 1))
            print('Total loss:' + str(plot_losses[-1]))
            print('note_loss:' + str(note_losses[-1]))
            print('\n')


if __name__ == '__main__':
    start()

