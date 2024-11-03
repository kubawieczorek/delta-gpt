from pathlib import Path

import torch

from jw.gpt.DataBatch import DataBatch
from jw.gpt.EncoderDecoder import EncoderDecoder
from jw.gpt.GPTLanuguageModel import GPTLanguageModel
from jw.gpt.Learning import Learning
from jw.gpt.cases.music.MusicConverter import MusicConverter

# hyperparameters
batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 512  # what is the maximum context length for predictions?
max_iters = 7000
eval_interval = 100
learning_rate = 2e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 768
n_head = 24
n_trans_blocks = 6
dropout = 0.1
split = 0.8
# ------------

print('device: ', device)

songs = []
folder = Path('maestro')
for file in folder.rglob('*.mid'):
    songs.append(file)
for file in folder.rglob('*.midi'):
    songs.append(file)

output_file = 'output-6-1.7.mid'

music_converter = MusicConverter(time_reduction=1,
                                 velocity_reduction=20,
                                 control_change_val_reduction=20,
                                 pitch_reduction=500,
                                 note_reduction=1
                                 )
music_keys = music_converter.extract_messages(songs)

print('total size: ', len(music_keys))
music_keys_unique = sorted(set(item for item in music_keys))
vocab_size = len(music_keys_unique)
print('vocab size: ', vocab_size)

encode_decoder = EncoderDecoder(music_keys_unique)
encoded_text = encode_decoder.encode(music_keys)

data_batch = DataBatch(encoded_text, block_size, batch_size, device, split)

model = GPTLanguageModel(vocab_size, n_embd, n_head, n_trans_blocks, block_size, dropout, device)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

learning = Learning(model, learning_rate, max_iters, eval_interval, eval_iters, data_batch)
learning.start()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_music_keys = encode_decoder.decodeToList(m.generate(context, max_new_tokens=1500)[0].tolist())
print(generated_music_keys)
music_converter.create_midi_from_flat_messages(generated_music_keys, output_file)

print(f"New MIDI file created with a flat list of messages preserving tracks: {output_file}")
