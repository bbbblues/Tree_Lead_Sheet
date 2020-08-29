import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle
import random
import matplotlib.pyplot as plt
import os

plt.switch_backend('agg')
from GT_chord_dict import *
from GT_train_sec_feed_note import save_path, train_size, CHDC_MAX_NUM

model_dir = save_path
gene_path = model_dir + '/gene/'
if not os.path.exists(gene_path):
    os.makedirs(gene_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_size = 127
chord_input_size = 158
pitch_input_size = 63
dur_input_size = 16
SEC_MAX_LENGTH = 6
CHI_MAX_LENGTH = 30
bar_line = '|'
beats_per_bar = 4
do_sample = True
chord_as_seed = True

chosen_seed = 126
chosen_model_epoch = '600'
gene_num = 30

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

    def forward_generation(self, sec, chord, note, hn, cn, encoder_outputs):
        # embedded_chord = F.relu(self.chord_embedding(chord).view(1, 1, -1))
        # embedded_pitch = F.relu(self.pitch_embedding(note[0]).view(1, 1, -1))
        # embedded_dur = F.relu(self.dur_embedding(note[1]).view(1, 1, -1))
        embedded_chord = self.chord_embedding(chord).view(1, 1, -1)
        embedded_pitch = self.pitch_embedding(note[0]).view(1, 1, -1)
        embedded_dur = self.dur_embedding(note[1]).view(1, 1, -1)
        embedded_sec = self.chord_embedding(sec).view(1, 1, -1)

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

        input = torch.cat([embedded_pitch, embedded_dur, embedded_chord, embedded_sec, context], -1)
        output, (hn, cn) = self.lstm(input, (hn, cn))
        pitch_out = self.pitch_out(output[0])
        # topv, topi = F.softmax(pitch_out, dim=-1).data.topk(1) ### if sampling, this need to change
        # generated_pitch_tensor = topi.squeeze().detach()
        pitch_prob = F.softmax(pitch_out, dim=-1)
        generated_pitch_tensor = torch.multinomial(pitch_prob[0], 1)
        generated_pitch = generated_pitch_tensor.item()
        embedded_generated_pitch = self.pitch_embedding(generated_pitch_tensor).view(1, -1)
        dur_out = self.dur_out(torch.cat([embedded_generated_pitch, output[0]], -1))
        dur_prob = F.softmax(dur_out, dim=-1)
        generated_dur = torch.multinomial(dur_prob[0], 1).item()
        # generated_pitch = torch.multinomial(pitch_prob[0], 1).item()
        return generated_pitch, generated_dur, (hn, cn)

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


def get_chord_train_set(train_size):
    # path = 'training_set/train_chords_POSTBOP.pickle'
    # path = '/home/peizhe/Weimar_training_set.pickle'
    path = '/Users/bbbblues/Desktop/GT_train_chords_new.pickle'
    train_set = pickle.load(open(path, 'rb'))
    return train_set[:train_size]


def get_seed(train_chords):
    '''
    seed_chords = []
    for sample in train_chords:
        for i, chord in enumerate(sample):
            if chord >= chords_index_dict['A1:']:
                seed_chords.append(sample[i:i+5])

    return seed_chords
    '''

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

    return seed_chords


def generate(encoder, section_decoder, chord_decoders, note_decoder,
             sec_max_length=SEC_MAX_LENGTH, child_max_length=CHI_MAX_LENGTH):
    with torch.no_grad():
        chord_train_set = get_chord_train_set(train_size)
        seeds = get_seed(chord_train_set)
        print(len(seeds))
        seed = seeds[chosen_seed]  ####### to be changed

        seed_tensor = torch.tensor(seed, dtype=torch.long, device=device).view(-1, 1)
        input_tensor = seed_tensor
        input_length = input_tensor.size()[0]

        encoder_hn = encoder.initHidden()
        encoder_cn = encoder.initHidden()

        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, (encoder_hn, encoder_cn) = encoder(
                input_tensor[ei], encoder_hn, encoder_cn)
            encoder_outputs[ei] = encoder_output[0, 0]

        section_input = torch.tensor([[chords_index_dict['<SOC>']]], device=device)

        section_hn = encoder_hn
        section_cn = encoder_cn

        decoded_chords = ['<SOC>']

        section_ind = 0
        while section_ind < SEC_MAX_LENGTH:
            # print(section_ind)
            section_output, (section_hn, section_cn) = section_decoder(section_input, section_hn, section_cn,
                                                                       encoder_outputs)
            # topv, topi = section_output.data.topk(1)
            topv, topi = F.softmax(section_output, dim=-1).data.topk(1)
            if topi.item() == chords_index_dict['<EOC>']:
                decoded_chords.append('<EOC>')
                break
            else:
                decoded_chords.append(index_chords_dict[topi.item()])
                # print('Section sybmols:')
                # print(index_chords_dict[topi.item()])

            if section_input.item() >= chords_index_dict['A:']:
                child_ind = 0
                child_hn = section_hn
                child_cn = section_cn
                child_input = topi.squeeze().detach()
                # print(child_input)

                if section_input.item() >= chords_index_dict['A:'] and section_input.item() <= chords_index_dict[
                    'A\'\'\':']:
                    chord_lstm = chord_decoders[0]
                elif section_input.item() >= chords_index_dict['B:'] and section_input.item() <= chords_index_dict[
                    'B\'\':']:
                    chord_lstm = chord_decoders[1]
                else:
                    chord_lstm = chord_decoders[2]

                while child_ind < CHI_MAX_LENGTH:
                    child_output, (child_hn, child_cn) = chord_lstm(child_input, child_hn, child_cn,
                                                                                     encoder_outputs)
                    child_ind += 1

                    if do_sample:
                        chd_prob = F.softmax(child_output, dim=-1)
                        chd_out_tensor = torch.multinomial(chd_prob[0], 1)
                    else:
                        topv_c, topi_c = F.softmax(child_output, dim=-1).data.topk(1)
                        chd_out_tensor = topi_c
                    decoded_chords.append(index_chords_dict[chd_out_tensor.item()])

                    if chd_out_tensor == chords_index_dict['<EOS>']:
                        section_input = torch.tensor([[chords_index_dict['<EOS>']]], device=device)
                        break
                    elif child_ind == CHI_MAX_LENGTH:
                        section_input = torch.tensor([[chords_index_dict['<EOS>']]], device=device)
                    else:
                        child_input = torch.tensor([[chd_out_tensor.item()]], device=device)

                section_ind += 1

            else:
                # section_input = torch.tensor([[chords_index_dict['<SOS>']]], device=device)
                section_input = topi.squeeze().detach()
                print(section_input)

        # Generate notes.
        note_start_symbol = torch.tensor([pitch_input_size - 1, dur_input_size - 1], device=device)
        note_input = note_start_symbol
        decoded_notes = []
        note_hn = note_decoder.initHidden()  # For reference before assignment?
        note_cn = note_decoder.initHidden()
        note_hn = encoder_hn
        note_ch = encoder_cn
        extra_beats = 0

        chords_len = len(decoded_chords)
        for i in range(chords_len):
            cur_chord = decoded_chords[i]
            cur_chord_ind = chords_index_dict[cur_chord]
            cur_chord_tensor = torch.tensor([[cur_chord_ind]], device=device)

            if i != (chords_len - 1):
                nxt_chord = decoded_chords[i + 1]
            if i != 0:
                lst_chord = decoded_chords[i - 1]

            if cur_chord_ind > chords_index_dict['|']:
                cur_sec_tensor = cur_chord_tensor
            elif cur_chord_ind < chords_index_dict['<SOC>']:
                if nxt_chord == bar_line or nxt_chord == '<EOS>':
                    if lst_chord == bar_line or lst_chord == '<SOS>':
                        total_beats = beats_per_bar
                    else:
                        total_beats = beats_per_bar / 2
                else:
                    total_beats = beats_per_bar / 2

                beat = extra_beats

                while (True):
                    if beat >= total_beats:
                        extra_beats = beat - total_beats
                        break

                    if do_sample:
                        pitch_ind, dur_ind, (note_hn, note_cn) = note_decoder.forward_generation(cur_sec_tensor,
                            cur_chord_tensor, note_input, note_hn, note_cn, encoder_outputs)
                    else:
                        pass

                    note_input = torch.tensor([pitch_ind, dur_ind], device=device)
                    '''
                    note_input = torch.zeros(pitch_input_size + dur_input_size)
                    note_input[pitch_ind] = 1
                    note_input[pitch_input_size + dur_ind] = 1
                    '''

                    decoded_pitch = pitch_ind + 36  #### the value 36 may change
                    decoded_beat = ind_beat_dict[dur_ind]
                    decoded_note = (decoded_pitch, decoded_beat)
                    decoded_notes.append(decoded_note)

                    beat += decoded_beat
                    # sys.std.out.flush()
                    if beat >= total_beats:
                        print(beat)
                        extra_beats = beat - total_beats
                        break

        return decoded_chords, decoded_notes


def generate_all(gene_num):
    test_enc = torch.load(model_dir + 'Enc_Epoch' + chosen_model_epoch + '.pickle')
    test_sec = torch.load(model_dir + 'Sec_Epoch' + chosen_model_epoch + '.pickle')
    test_chords = [
        torch.load(model_dir + 'Chord' + str(i) + '_Epoch' + chosen_model_epoch + '.pickle') for i
        in range(CHDC_MAX_NUM)]
    test_note = torch.load(model_dir + 'Note_Epoch' + chosen_model_epoch + '.pickle')

    for i in range(gene_num):
        chords, notes = generate(test_enc, test_sec, test_chords, test_note)
        pickle.dump(chords, open(gene_path + 'grtd_chords_' + chosen_model_epoch +
                                 '_seed' + str(chosen_seed) + '_' + str(i) +'.pickle', 'wb'))
        pickle.dump(notes, open(gene_path + 'grtd_notes_' + chosen_model_epoch + '_seed' +
                                str(chosen_seed) + '_' + str(i) + '.pickle', 'wb'))


if __name__ == '__main__':
    generate_all(gene_num)



