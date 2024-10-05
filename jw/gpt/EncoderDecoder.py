# create a mapping from characters to integers
class EncoderDecoder:
    def __init__(self, chars):
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
        self.decodeToStr = lambda l: ''.join([itos[i] for i in l])
        self.decodeToList = lambda l: [itos[i] for i in l]
