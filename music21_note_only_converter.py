

# from music21 import converter, note, stream, meter, tempo, harmony
from music21 import *
import re
from GT_chord_dict import *
import pickle
import os
# from GT_generation import chosen_model_epoch, chosen_seed, gene_path, gene_num
# from one_layer_generation import chosen_model_epoch_ol, chosen_seed_ol, gene_path, gene_num_ol
from GT_grt_sec_feed_note import chosen_model_epoch, chosen_seed, gene_path, gene_num

# to be changed
# dir_path = 'eval/notes_only_model/epoch' + chosen_model_epoch_ol + '/seed' + str(chosen_seed_ol) + '/'
dir_path = 'eval/notes/sec_feed_note/epoch' + chosen_model_epoch + '/seed' + str(chosen_seed) + '/'
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

def note_converter(pitch_cor_notes, smp_ind):
    s = stream.Score()
    m = meter.TimeSignature('4/4')

    s_n = stream.Part()
    s_n.insert(0, m)
    # s_n.insert(0, instrument.Saxophone())

    for item in pitch_cor_notes:
        if item[0] == 'rest':
            note_obj = note.Rest()
            note_obj.duration.quarterLength = item[1]
        else:
            note_obj = note.Note(item[0])
            note_obj.duration.quarterLength = item[1]
        s_n.append(note_obj)

    s.insert(0, s_n)
    generated_notes = s.write('midi', fp=dir_path+'smp'+str(smp_ind)+'_'+'.midi')


def convert():
    for smp_i in range(gene_num):
        generated_notes_dir = gene_path + 'grtd_notes_' + chosen_model_epoch + '_seed' + str(chosen_seed) + '_' + str(smp_i) + '.pickle'
        generated_notes = pickle.load(open(generated_notes_dir, 'rb'))

        pitch_cor_notes = []
        for item in generated_notes:
            pitch_cor_notes.append((midi_to_pitch[item[0]], item[1]))

        note_converter(pitch_cor_notes, smp_i)


if __name__ == '__main__':
    convert()


