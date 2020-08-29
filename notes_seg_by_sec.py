'''
Convert generated music data format to xml or midi files in terms of section boundaries
instead of converting the whole music piece.
'''

# from music21 import converter, note, stream, meter, tempo, harmony
from music21 import *
import re
# from chord_dict import *
from GT_chord_dict import *
import pickle
import os
# from generation_attn import chosen_model_epoch, chosen_seed, model_dir
from GT_generation import chosen_model_epoch, chosen_seed, gene_path, gene_num

# harmony.ChordSymbol.writeAsChord = True

# to be changed
dir_path = 'eval/notes_seg/epoch' + chosen_model_epoch + '/seed' + str(chosen_seed) + '/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

midi_to_pitch = {}
for midi in range(36, 98):

    octave = str(int(midi / 12) - 1)

    if midi % 12 == 0:
        midi_to_pitch[midi] = 'C' + octave
    elif midi % 12 == 1:
        midi_to_pitch[midi] = 'C#' + octave
    elif midi % 12 == 2:
        midi_to_pitch[midi] = 'D' + octave
    elif midi % 12 == 3:
        midi_to_pitch[midi] = 'D#' + octave
    elif midi % 12 == 4:
        midi_to_pitch[midi] = 'E' + octave
    elif midi % 12 == 5:
        midi_to_pitch[midi] = 'F' + octave
    elif midi % 12 == 6:
        midi_to_pitch[midi] = 'F#' + octave
    elif midi % 12 == 7:
        midi_to_pitch[midi] = 'G' + octave
    elif midi % 12 == 8:
        midi_to_pitch[midi] = 'G#' + octave
    elif midi % 12 == 9:
        midi_to_pitch[midi] = 'A' + octave
    elif midi % 12 == 10:
        midi_to_pitch[midi] = 'A#' + octave
    elif midi % 12 == 11:
        midi_to_pitch[midi] = 'B' + octave

midi_to_pitch[98] = 'rest'


def nt_seg_converter(generated_chords, pitch_cor_notes, smp_ind):
    # find section boundaries
    acu_beat = 0
    sec_bd = []
    for i, item in enumerate(generated_chords):
        if chords_index_dict[item] < chords_index_dict['<SOC>']:
            prev_chord = generated_chords[i - 1]
            next_chord = generated_chords[i + 1]
            if chords_index_dict[prev_chord] >= chords_index_dict['<SOC>'] and chords_index_dict[next_chord] >= \
                    chords_index_dict['<SOC>']:
                acu_beat += 4
            else:
                acu_beat += 2
        elif chords_index_dict[item] > chords_index_dict['A:']:
            sec_bd.append(acu_beat)

    sec_bd.append(10000)
    acu_beat_n = 0
    cur_ind = 0
    cur_bd = sec_bd[cur_ind]

    s = stream.Score()
    m = meter.TimeSignature('4/4')
    s_n = stream.Part()
    s_n.insert(0, m)
    #s_n.insert(0, instrument.Saxophone())

    for item in pitch_cor_notes:
        if item[0] == 'rest':
            note_obj = note.Rest()
            note_obj.duration.quarterLength = item[1]
        else:
            note_obj = note.Note(item[0])
            note_obj.duration.quarterLength = item[1]
        s_n.append(note_obj)
        acu_beat_n += item[1]
        if acu_beat_n >= cur_bd:
            s.insert(0, s_n)
            gene_piece = s.write('midi', fp=dir_path+'smp'+str(smp_ind)+'_'+str(cur_ind)+'.midi')
            s = stream.Score()
            m = meter.TimeSignature('4/4')
            s_n = stream.Part()
            s_n.insert(0, m)
            #s_n.insert(0, instrument.Saxophone())
            if cur_ind < len(sec_bd)-1:
                cur_ind += 1
                cur_bd = sec_bd[cur_ind]

    s.insert(0, s_n)
    gene_piece = s.write('midi', fp=dir_path+'smp'+str(smp_ind)+'_'+str(cur_ind)+'.midi')


def convert():
    for smp_i in range(gene_num):
        generated_notes_dir = gene_path + 'grtd_notes_' + chosen_model_epoch + '_seed' + str(chosen_seed) + '_' + str(smp_i) + '.pickle'
        generated_notes = pickle.load(open(generated_notes_dir, 'rb'))
        generated_chords_dir = gene_path + 'grtd_chords_' + chosen_model_epoch + '_seed' + str(chosen_seed) + '_' + str(smp_i) +'.pickle'
        generated_chords = pickle.load(open(generated_chords_dir, 'rb'))

        pitch_cor_notes = []
        for item in generated_notes:
            pitch_cor_notes.append((midi_to_pitch[item[0]], item[1]))

        nt_seg_converter(generated_chords, pitch_cor_notes, smp_i)


if __name__ == '__main__':
    convert()
