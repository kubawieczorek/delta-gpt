import torch


class DataBatch:
    def __init__(self, encoded_text, block_size, batch_size, device, split=0.9):
        data = torch.tensor(encoded_text, dtype=torch.long)
        n = int(split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch_rnd(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
