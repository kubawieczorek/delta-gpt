import torch
import re

from jw.gpt.DataBatch import DataBatch
from jw.gpt.EncoderDecoder import EncoderDecoder
from jw.gpt.GPTLanuguageModel import GPTLanguageModel
from jw.gpt.Learning import Learning

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 300  # what is the maximum context length for predictions?
max_iters = 4000
eval_interval = 200
learning_rate = 10e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 300
n_head = 8
n_trans_blocks = 6
dropout = 0.1
# ------------

torch.manual_seed(1337)

with open('input/input_conv_3.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Directory containing the text files
directory_path = 'input'

#words = sorted(list(set(re.findall(r'\s|\w+|[^\w\s]', text))))
words = sorted(list(set(text)))
vocab_size = len(words)

print('vocab size', vocab_size)

encode_decoder = EncoderDecoder(words)
encoded_text = encode_decoder.encode(text)

data_batch = DataBatch(encoded_text, block_size, batch_size, device)

model = GPTLanguageModel(vocab_size, n_embd, n_head, n_trans_blocks, block_size, dropout, device)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

learning = Learning(model, learning_rate, max_iters, eval_interval, eval_iters, data_batch)
learning.start()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
open('output_1.txt', 'w', encoding='utf-8').write(
    encode_decoder.decodeToStr(m.generate(context, max_new_tokens=2000)[0].tolist()))
