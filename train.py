from torch import nn
from torch import optim


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
            y_pred = model(inputs)
            if loss_mode == 1:
                loss = criterion(y_pred, labels)
            if loss_mode == 2:
                # penalize by the prefix length
                loss = criterion(y_pred, labels) * (len(y_pred.nonzero())) / 100
            if loss_mode == 3:
                # penalize by 1/100 of the prefix length
                loss = criterion(y_pred, labels) * len(y_pred.nonzero())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == len(train_loader) - 1:
                print('Average loss at epoch {}: {}'.format(epoch, sum(losses) / i))
    print('Training complete.')
    return model
