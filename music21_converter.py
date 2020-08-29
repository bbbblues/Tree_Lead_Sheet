

# from music21 import converter, note, stream, meter, tempo, harmony
from music21 import *
import re
# from chord_dict import *
from GT_chord_dict import *
import pickle
import os
# from generation_attn import chosen_model_epoch, chosen_seed, model_dir
from GT_grt_sec_feed_note import chosen_model_epoch, chosen_seed, gene_path

# chosen_epoch = '900'
# chosen_seed = '18'
harmony.ChordSymbol.writeAsChord = True
chosen_sample = '_29'

# to be changed
dir_path = 'generation_result/GT_sec_feed_note/epoch' + chosen_model_epoch + '/seed' + str(chosen_seed) + '/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

generated_notes_dir = gene_path + 'grtd_notes_' + chosen_model_epoch + '_seed' + str(chosen_seed) + chosen_sample + '.pickle'
generated_notes = pickle.load(open(generated_notes_dir, 'rb'))
generated_chords_dir = gene_path + 'grtd_chords_' + chosen_model_epoch + '_seed' + str(chosen_seed) + chosen_sample + '.pickle'
generated_chords = pickle.load(open(generated_chords_dir, 'rb'))
print(generated_chords)

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

pitch_cor_notes = []
for item in generated_notes:
    pitch_cor_notes.append((midi_to_pitch[item[0]], item[1]))

def all_converter(generated_chords, pitch_cor_notes):
    s = stream.Score()
    m = meter.TimeSignature('4/4')

    s_c = stream.Part()
    s_c.insert(0, m)
    s_c.insert(0, instrument.Piano())

    for i, item in enumerate(generated_chords):
        if chords_index_dict[item] < chords_index_dict['<SOC>']:
            item = re.sub('b', '-', item)
            # m7b5 no need to be changed
            item = re.sub('m7-5', 'm7b5', item)
            chord_obj = harmony.ChordSymbol(item)
            prev_chord = generated_chords[i - 1]
            next_chord = generated_chords[i + 1]
            if chords_index_dict[prev_chord] >= chords_index_dict['<SOC>'] and chords_index_dict[next_chord] >= \
                    chords_index_dict['<SOC>']:
                chord_obj.duration.quarterLength = 4
            else:
                chord_obj.duration.quarterLength = 2
            s_c.append(chord_obj)
            # s_c.append(expressions.TextExpression(item))
        elif chords_index_dict[item] >= chords_index_dict['A:']:
            s_c.append(expressions.RehearsalMark(item))

    s_n = stream.Part()
    s_n.insert(0, m)
    s_n.insert(0, instrument.Saxophone())

    for item in pitch_cor_notes:
        if item[0] == 'rest':
            note_obj = note.Rest()
            note_obj.duration.quarterLength = item[1]
        else:
            note_obj = note.Note(item[0])
            note_obj.duration.quarterLength = item[1]
        s_n.append(note_obj)

    s.insert(0, s_n)
    s.insert(0, s_c)
    return s


generated_score = all_converter(generated_chords, pitch_cor_notes)
f_midi = generated_score.write('midi', fp= dir_path + chosen_sample + '.midi')
f_xml = generated_score.write('musicxml', fp= dir_path + chosen_sample + '.xml')



