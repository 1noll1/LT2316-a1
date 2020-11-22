import argparse
from glob import glob
import torch
from GRUclassifier import GRUclassifier
from prefixloader import PrefixLoader
from torch.utils.data import Dataset, DataLoader
from train import train_model
import pandas as pd
import pickle

def loadfiles(directory):
    # sorted to make sure we get x_test, x_train, y_test, y_train
    # split on sentences
    print('directory:', directory)
    for f in sorted(glob(directory + '[xy]_*.txt')):
        print('loading file', f)
    return [open(f, 'r').read().split('\n') for f in sorted(glob(directory + '[xy]*.txt'))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GRU model.")
    parser.add_argument("directory", type=str,
                        help="The directory containing the test and training files")
    parser.add_argument("-batch_size", type=int, default=16, help="The desired mini batch size")
    parser.add_argument("-E", "--num_epochs", type=int, default=10, help="The desired amount of epochs for training")
    parser.add_argument('-l', '--langs', nargs='+', help='list of languages to train on', default=['ukr', 'rus', 'bul', 'swe', 'eng', 'nno', 'pol', 'bel', 'ang', 'rue'])
    parser.add_argument("--loss_mode", type=int, default=1, help="Which of the 3 loss modes to use– see README")

    args = parser.parse_args()

    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")
    print(f'Using {dev}')

    _, x_train, _, y_train = loadfiles(args.directory)
    labels = pd.read_csv('wili-2018/labels.csv', sep=';', index_col=0)
    print('Using languages:' + '\n')
    for index, row in labels.iterrows():
        if index in args.langs:
            print(row['English'])

    if not args.langs:
        # default languages
        print('Using default languages')

    dataset = PrefixLoader(args.langs, x_train, y_train, dev,)
    pickle.dump(dataset, open("train_dataset.pkl", "wb"))
    print('Saving train dataset to train_dataset.pkl')
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print('Training model with batch size {}'.format(args.batch_size))
    print('Using loss mode {}'.format(args.loss_mode))
    model = GRUclassifier(dataset.vocab_size, len(dataset.x_train[0]), 50, dataset.num_classes, dev=dev)
    trained_model = train_model(model, args.num_epochs, dev, train_loader=train_loader, loss_mode=args.loss_mode)
    model_name = 'trained_model_' + 'e' + str(args.num_epochs) + 'b' + str(args.batch_size) + 'l' + str(args.loss_mode)
    print(f'Saving model as {model_name}')
    torch.save(trained_model, model_name)
