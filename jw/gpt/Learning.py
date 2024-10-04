import torch


class Learning:
    def __init__(self, model, learning_rate, max_iters, eval_interval, eval_iters, data_batch):
        self.model = model
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.data_batch = data_batch
        # create a PyTorch optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def start(self):
        for iter in range(self.max_iters):
            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = self.data_batch.get_batch_rnd('train')

            # evaluate the loss
            logits, loss = self.model.forward(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.data_batch.get_batch_rnd(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.model.train()
        return out
