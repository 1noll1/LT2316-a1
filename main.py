from prefixloader import PrefixLoader
import os
from GRUclassifier import GRUclassifier
from GRUClassifier_nobatch import GRUclassifier_nb
import argparse
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
from train import trained_batches
from train_nobatch import train_wobatch
from torch import optim

def loadfiles(directory):
    # sorted to make sure we get x_test, x_train, y_test, y_train
    # split on sentences
    for f in sorted(glob(directory + '[xy]_*.txt')):
        print('loading file', f)
    return [open(f, 'r').read().split('\n') for f in sorted(glob(directory + '[xy]*.txt'))]    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train a GRU model with minibathes.")
    parser.add_argument("--directory", required=True, type=str, help="The directory containing the test and training files")
    parser.add_argument("--batch_size", type=int, default=200, help="The desired mini batch size")
    parser.add_argument("-E", "--num_epochs", type=int, default=20, help="The desired amount of epochs for training")
    parser.add_argument("--eval", type=bool, default=False, help="Evaluate the trained model")

    args = parser.parse_args()

    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")

    x_test, x_train, y_test, y_train = loadfiles(args.directory)
    dataset = PrefixLoader(['ukr', 'rus', 'bul', 'swe', 'eng', 'nnol', 'pl', 'bel', 'ang', 'rue'], x_train, y_train, dev)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.batch_size > 1:
        print('Training model with {} batches'.format(args.batch_size))
        model = GRUclassifier(dataset.vocab_size, len(dataset.x_train[0]), 50, dataset.num_classes, dev=dev)
        trained_model = trained_batches(model, args.num_epochs, dev, train_loader=train_loader, loss_mode=1)
        torch.save(trained_model, 'trained_model_' + 'e' + str(args.num_epochs) + 'b' + str(args.batch_size))
    if args.batch_size == 1:
        print('Training model without batches')
        model = GRUclassifier_nb(dataset.vocab_size, len(dataset.x_train[0]), 50, dataset.num_classes, dev=dev)
        trained_model = train_wobatch(model, dataset, args.num_epochs, dev, train_loader=train_loader, loss_mode=1)
        torch.save(trained_model, 'trained_model_' + 'e' + str(args.num_epochs))
    if args.eval:
        from eval_script import model_eval
        dataset = PrefixLoader(['ukr', 'rus', 'bul', 'swe', 'eng', 'nnol', 'pl', 'bel', 'ang', 'rue'], x_test, y_test, dev)
        train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
        model_eval(test_loader, trained_model, eval_mode=1)
    
    

