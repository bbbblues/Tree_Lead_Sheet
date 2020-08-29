
import sqlite3
import re
import sys
import pickle
from chord_dict import *

file = '/Users/bbbblues/Desktop/wjazzd.db'
weimar = sqlite3.connect(file)

c_sig = weimar.cursor()
c_sig.execute('SELECT signature FROM solo_info')
sigs = c_sig.fetchall()

err_sig_ind = []
for i in range(len(sigs)):
    if sigs[i][0] != '4/4':
        err_sig_ind.append(i)

# select some specific style for a cleaner dataset
selected_style = 'POSTBOP'
c_style = weimar.cursor()
c_style.execute('SELECT style FROM solo_info')
styles = c_style.fetchall()

selected_style_ind = []
for i in range(len(styles)):
    if styles[i][0] == selected_style:
        selected_style_ind.append(i)
        
#print(err_sig_ind)
print(selected_style_ind)

c_chords = weimar.cursor()
c_chords.execute('SELECT chord_changes FROM solo_info')
all_chords = c_chords.fetchall()

training_set_chords = []
for item in all_chords:
    
    chords = ['<SOC>']
    symbol = ''
    item = item[0]

    for i, char in enumerate(item):
        if char != ' ' and char != "\n":
            if char == '|' and item[i-1] == ' ' and item[i+1] == '|':
                if (i+1) == len(item)-1 or item[i+2] == '\n':
                    chords.append('<EOS>')
                else:
                    chords.append('<SOS>')
            elif char == '|' and item[i-1] == ' ':
                chords.append('|')
            elif char != '|':
                symbol += char
        elif char == ' ' and symbol != '':
            chords.append(symbol)
            symbol = ''
        
    chords.append('<EOC>')

    for i, chord in enumerate(chords):
        chords[i] = re.sub(r'-', 'm', chords[i])
        #chords[i] = re.sub(r'o', '', chords[i])
        #chords[i] = re.sub(r'+', '', chords[i])
        chords[i] = re.sub(r'j', 'maj', chords[i])
        chords[i] = re.sub(r'alt', '', chords[i])
        chords[i] = re.sub(r'9#11#', '', chords[i])
        chords[i] = re.sub(r'911#', '', chords[i])
        chords[i] = re.sub(r'911', '', chords[i])
        chords[i] = re.sub(r'913', '', chords[i])    
        chords[i] = re.sub(r'9#', '', chords[i])
        chords[i] = re.sub(r'9b', '', chords[i])
        chords[i] = re.sub(r'9', '', chords[i])
        chords[i] = re.sub(r'13', '', chords[i])
        chords[i] = re.sub(r'sus', '', chords[i])
        chords[i] = re.sub(r'/\w#', '', chords[i])
        chords[i] = re.sub(r'/\wb', '', chords[i])
        chords[i] = re.sub(r'/\w', '', chords[i])
        chords[i] = re.sub(r'Db', 'C#', chords[i])
        chords[i] = re.sub(r'F#', 'Gb', chords[i])
        chords[i] = re.sub(r'G#', 'Ab', chords[i])

    training_set_chords.append(chords)
    
for i in range(len(training_set_chords)):
    song = training_set_chords[i]
    for j in range(len(song)):
        if song[j] == '<SOS>':
            song[j] = song[j-1]
            song[j-1] = '<SOS>'
    training_set_chords[i] = tuple(song)


chord_set = []
err_chord_ind = []


for i, item in enumerate(training_set_chords):
    if i not in err_sig_ind and i in selected_style_ind:
        try:
            chord_set.append([chords_index_dict[chord] for chord in item])
        except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
            exception_str = 'Unexpected error in ' + str(i)  + ':\n', e, sys.exc_info()[0]
            err_chord_ind.append(i)
            print(exception_str)

# save_train_chord_path = 'chord_set/train_chords.pickle'
save_train_chord_path = 'training_set/train_chords_' + selected_style + '.pickle'
pickle.dump(chord_set, open(save_train_chord_path, 'wb'))
# save_err_chord_ind_path = ''
# pickle.dump(err_chord_ind, open(save_err_chord_ind_path, 'wb'))

# the notes
c = weimar.cursor()
notes_orig = []
for i in range(1, 457):
    c.execute("SELECT pitch,onset,duration,beatdur,bar,beat,tatum FROM melody WHERE melid='" + str(i) + "'")
    notes_orig.append(c.fetchall())

all_notes = []

for i, song in enumerate(notes_orig):
    if i not in err_chord_ind and i not in err_sig_ind and i in selected_style_ind:
        song_notes = []
        for j, note in enumerate(song):
            pitch = note[0]
            duration = note[2]
            beatdur = note[3]
            bar = note[4]
            beat = note[5]
            tatum = note[6]
            
            if bar <= 0:
                continue
            if j == len(song)-1:
                dur = duration / beatdur
                song_notes.append((pitch, dur))
            else:
                if len(song_notes) == 0 and not(beat == 1 and tatum == 1):
                    # the first note appended, check if rest before it
                    pitch0 = 98
                    if song[j+1][5] != beat:
                        dur0 = (beat-1) + (tatum-1) / 2
                    else:
                        dur0 = (beat-1) + (tatum-1) / 4
                    song_notes.append((pitch0, dur0))
                    
                onsets_dif = song[j+1][1] - song[j][1]
                if onsets_dif - duration > beatdur:
                    pitch1 = note[0]
                    dur1 = duration / beatdur
                    song_notes.append((pitch1, dur1))
                
                    pitch2 = 98 #### pitch 36-97, plus silence 98, transferred to 0-62   
                    dur2 = (onsets_dif - duration/2) / beatdur ## plolong rest to compensate
                    song_notes.append((pitch2, dur2))
                else:
                    dur = onsets_dif / beatdur
                    song_notes.append((pitch, dur))
        all_notes.append(song_notes)


beat_list = [0.0625, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
beat_cor_all_notes = []

for song in all_notes:
    new_song = []
    for note in song:
        pitch = note[0]
        dur = note[1]
        min = 100
        min_ind = 100
        for i in range(len(beat_list)):
            if abs(beat_list[i] - dur) < min:
                min = abs(beat_list[i] - dur)
                min_ind = i
        new_dur = beat_list[min_ind]
        new_song.append((pitch, new_dur))
    beat_cor_all_notes.append(new_song)

# save_train_note_path = 'training_set/train_notes.pickle'
save_train_note_path = 'training_set/train_notes_' + selected_style + '.pickle'
pickle.dump(beat_cor_all_notes, open(save_train_note_path, 'wb'))

# print(len(chord_set))
# print(len(beat_cor_all_notes))

# print(chord_set[-1])
# print(beat_cor_all_notes[-1])




