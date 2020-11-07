import torch
from torch import nn

criterion = nn.NLLLoss()


class GRUclassifier(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, dev):
        super().__init__()
        self.dev = dev
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        batch_size = x.shape[0]
        output = self.embed(x)
        output = output.contiguous().view(batch_size, 10000, -1)
        h = self.init_hidden(len(x))
        output, hidden = self.gru(output, h)
        output = self.linear(output)  # squish it! This is a fully connected layer!
        return self.logsoftmax(output)

    def set_dev(self, dev):
        self.dev = dev

    def init_hidden(self, batch_len):
        #return torch.zeros(8, self.num_layers, self.hidden_size).to(self.dev)
        #return torch.zeros(self.num_layers, 8, self.hidden_size).to(self.dev)
        return torch.zeros(self.num_layers, batch_len, self.hidden_size).to(self.dev)
