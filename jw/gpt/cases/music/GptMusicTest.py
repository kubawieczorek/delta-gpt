from pathlib import Path

from jw.gpt.EncoderDecoder import EncoderDecoder
from jw.gpt.cases.music.MusicConverter import mid_to_notes, create_midi

songs = []
folder = Path('songs2')
for file in folder.rglob('*.mid'):
    songs.append(file)

notes = mid_to_notes(songs)

pitch_names = sorted(set(item for item in notes))
vocab_size = len(pitch_names)
print('vocab size: ', vocab_size)

encode_decoder = EncoderDecoder(pitch_names)
encoded_text = encode_decoder.encode(notes)

music = encode_decoder.decodeToList(encoded_text)
create_midi(notes)
