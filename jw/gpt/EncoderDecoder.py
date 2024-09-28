# create a mapping from characters to integers
class EncoderDecoder:
    def __init__(self, chars):
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
