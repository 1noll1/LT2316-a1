import random
import torch
from torch import nn
from torch import optim

def train_wobatch(model, dataset, num_epochs, dev, train_loader, loss_mode=1):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    model = model.to(dev)
    model.set_dev(dev)
    
    for epoch in range(1, num_epochs+1):
        random.shuffle(dataset.pairs)
        print('starting epoch {}'.format(epoch))
        losses = 0
        i = 0
        for prefix, label in dataset.pairs:
            i += 1
            prefix = torch.LongTensor(prefix).to(dev)
            label = torch.LongTensor([label]).to(dev)
            y_pred = model(prefix)
            if loss_mode == 1:
                loss = criterion(y_pred, label)
            if loss_mode == 2:
                loss = criterion(y_pred, label) * (len(y_pred.nonzero()/100))
            if loss_mode == 3:
                loss = criterion(y_pred, label) * len(y_pred.nonzero())
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == len(dataset.pairs):
                print('Average loss at epoch {}: {}'.format(epoch, losses/i))
    print('Training complete.')
    return model
