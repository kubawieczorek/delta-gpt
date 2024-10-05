from pathlib import Path

import torch

from jw.gpt.DataBatch import DataBatch
from jw.gpt.EncoderDecoder import EncoderDecoder
from jw.gpt.GPTLanuguageModel import GPTLanguageModel
from jw.gpt.Learning import Learning
from jw.gpt.cases.music.MusicConverter import mid_to_notes, create_midi

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 16  # what is the maximum context length for predictions?
max_iters = 400
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 32
n_head = 6
n_trans_blocks = 6
dropout = 0.3
# ------------

print('device: ', device)

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

data_batch = DataBatch(encoded_text, block_size, batch_size, device)

model = GPTLanguageModel(vocab_size, n_embd, n_head, n_trans_blocks, block_size, dropout, device)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

learning = Learning(model, learning_rate, max_iters, eval_interval, eval_iters, data_batch)
learning.start()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
music = encode_decoder.decodeToList(m.generate(context, max_new_tokens=150)[0].tolist())
create_midi(music)
