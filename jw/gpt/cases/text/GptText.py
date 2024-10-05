import torch

from jw.gpt.DataBatch import DataBatch
from jw.gpt.EncoderDecoder import EncoderDecoder
from jw.gpt.GPTLanuguageModel import GPTLanguageModel
from jw.gpt.Learning import Learning

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 384
n_head = 6
n_trans_blocks = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('input2.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

encode_decoder = EncoderDecoder(chars)
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
print(encode_decoder.decodeToStr(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
