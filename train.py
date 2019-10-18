parser = argparse.ArgumentParser(description="Train a maximum entropy model.")
parser.add_argument("-X", "--x_file", metavar="X", dest="x_file", type=str, help="File containing x-data")
parser.add_argument("-Y", "--y_file", metavar="Y", dest="y_file", type=str, help="File containing the target data")


parser.add_argument("-B", "--batches", metavar="B", dest="batches", type=int, default=500, help="The desired amount of batches")
parser.add_argument("-H", "--hidden", metavar="H", dest="hidden", type=int, default=50 help="The desired hidden layers size")
parser.add_argument("modelfile", type=str,
                    help="The filname to which you wish to write the trained model.")

if args.filename.endswith('.py'):
    filename = args.filename
else:
    filename = args.filename + '.py'

def trained_batches(num_epochs):
    model = GRUclassifier(dataset.vocab_size, 100, args.hidden, 3)
    batch_len = args.batches
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    model = model.to(dev)
    model.set_dev(dev)

    for epoch in range(1, num_epochs+1):
        losses = []
        print('Starting epoch...')
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)      
            print('Epoch no.', epoch, 'instance no.:', i, 'loss:', loss.item())  
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == len(train_loader)-1:
                print('Average loss at epoch {}: {}'.format(epoch, sum(losses)/i))
    print('Training complete.')
    return model

print('Training model')
model = saved_batches(num_epochs)
print('Saving model')
torch.save(model, filename)
print('Saving model to file')