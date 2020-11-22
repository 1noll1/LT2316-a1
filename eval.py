import argparse
import torch
from main import loadfiles
from prefixloader import PrefixLoader
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import pickle


def model_eval(test_loader, model, eval_mode=1):
    model = model.eval()
    a = 0  # accurate predictions
    predictions = {'Instance {}'.format(i): 0 for i in range(1, len(test_loader) + 1)}
    failures = 0
    scored = 0
    total = 0  # of the instances that are not complete failures
    print('model', model)

    with torch.no_grad():
        for sents, labels in tqdm(test_loader):
            a += 1
            outputs = model(sents)
            _, predicted = torch.max(outputs.data, 2)
            predicted = predicted.view(100, 100)

            if eval_mode == 1:

                for i, label in enumerate(labels):
                    # the true label is the same for every prefix
                    label = label[0]
                    for ix, predicted_label in enumerate(predicted[i]):
                        if label == predicted_label:
                            predictions['Instance {}'.format(a)] += 1
                            print('Correct guess at prefix length {}'.format(ix))
                            break
                        else:
                            pass

            elif eval_mode == 2:
                for i, predicted_labels in enumerate(predicted):
                    for ix, label in enumerate(predicted_labels):
                        if label != labels[0][0]:
                            if ix == 99:
                                print('Complete failure!')
                                failures += 1
                        elif label == labels[0][0]:
                            print('Number of characters until hit score: {}'.format(ix + 1))
                            scored += (ix + 1)
                            total += 1
                            break

    if eval_mode == 1:
        print('Number of accurate guesses per sentence')
        for key in predictions:
            print(key, ':', predictions[key])

        accurate = 0

        print('Overall accuracy:')
        for key in predictions:
            if predictions[key] > 0:
                accurate += 1
        print(accurate / len(predictions) * 100, '%')

    if eval_mode == 2:
        print('Total amount of complete failures: {}'.format(failures))
        print('Average number of characters until hit score: {}'.format(round(scored / total)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate GRU model.")

    parser.add_argument("directory", type=str, default='wili-2018/',
                        help="The directory containing the test and training files")
    parser.add_argument("--modelfile", type=str, required=True, help="Name of the file containing the trained model")
    parser.add_argument('-l', '--langs', nargs='+', help='list of languages to train evaluate on')
    parser.add_argument('--eval_mode', type=int, required=True,
                        help="Which of the 2 eval modes to implement – see README")
    parser.add_argument('--cuda', type=str, help="Specify GPU")

    args = parser.parse_args()

    labels = pd.read_csv('wili-2018/labels.csv', sep=';', index_col=0)
    print('Using languages:')
    for index, row in labels.iterrows():
        if index in args.langs:
            print(row['English'])

    trained_model = torch.load(args.modelfile)
    x_test, _, y_test, _ = loadfiles(args.directory)
    dev = torch.device(f'cuda:{args.cuda}')
    # print('Using', dev)

    train_dataset = pickle.load("train_dataset.pkl")
    test_dataset = PrefixLoader(args.langs, x_test, y_test, dev, train_dataset)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    model_eval(test_loader, trained_model, args.eval_mode)
