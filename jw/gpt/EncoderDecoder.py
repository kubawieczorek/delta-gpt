# create a mapping from characters to integers
class EncoderDecoder:
    def __init__(self, words):
        stoi = {ch: i for i, ch in enumerate(words)}
        itos = {i: ch for i, ch in enumerate(words)}
        self.encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
        self.decodeToStr = lambda l: ''.join([itos[i] for i in l])
        self.decodeToList = lambda l: [itos[i] for i in l]
