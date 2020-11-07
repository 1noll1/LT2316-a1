from torch import nn
from torch import optim
import torch


def train_model(model, num_epochs, dev, train_loader, loss_mode=1):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    model = model.to(dev)
    model.set_dev(dev)

    for epoch in range(1, num_epochs + 1):
        losses = []
        print('starting epoch...')
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            #labels = torch.LongTensor([[label] * 100 for label in labels])
            y_pred = model(inputs)

            batch_size, _, output_size = y_pred.shape
            y_pred = y_pred.view(batch_size, 100, 100, -1)

            running_loss = 0

            for ix, prefix in enumerate(y_pred[:, 1]):
                print('prefix', ix)
                # print(prefix)
                # print('prefix shape:', prefix.shape)
                if loss_mode == 1:
                    loss = criterion(prefix, labels[ix])
                if loss_mode == 2:
                    # penalize by the prefix length
                    loss = criterion(prefix, labels[ix]) * (len(prefix.nonzero())) / 100
                if loss_mode == 3:
                    # penalize by 1/100 of the prefix length
                    loss = criterion(prefix, labels[ix]) * len(prefix.nonzero())
                losses.append(loss.item())
                running_loss += loss
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            if i == len(train_loader) - 1:
                print('Average loss at epoch {}: {}'.format(epoch, sum(losses) / i))
    print('Training complete.')
    return model
