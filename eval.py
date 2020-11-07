import argparse
import torch
from main import loadfiles
from prefixloader import PrefixLoader
from torch.utils.data import DataLoader
import pandas as pd


def model_eval(test_loader, model, eval_mode=1):
    model = model.eval()
    a = 0  # accurate predictions
    predictions = {'Instance {}'.format(i): 0 for i in range(1, len(test_loader) + 1)}
    failures = 0
    scored = 0
    total = 0  # of the instances that are not complete failures

    with torch.no_grad():
        for x, y in test_loader:
            a += 1
            x = x.to('cpu')
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            if eval_mode == 1:
                for i, value in enumerate(y):
                    value = value.to('cpu')
                    if value == predicted[i]:
                        predictions['Instance {}'.format(a)] += 1
                        print('Correct guess at prefix length {}'.format(i))
                    else:
                        continue
            elif eval_mode == 2:
                for i, value in enumerate(predicted):
                    if value != y[i]:
                        if i == len(predicted) - 1:
                            print('Complete failure!')
                            failures += 1
                        else:
                            pass
                    elif value == y[i]:
                        print('Number of characters until hit score: {}'.format(i + 1))
                        scored += (i + 1)
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

    parser.add_argument("--directory", type=str, required=True,
                        help="The directory containing the test and training files")
    parser.add_argument("--modelfile", type=str, required=True, help="Name of the file containing the trained model")
    parser.add_argument('-l', '--langs', nargs='+', help='list of languages to train evaluate on')
    parser.add_argument('--eval_mode', type=int, required=True,
                        help="Which of the 2 eval modes to implement – see README")

    args = parser.parse_args()

    labels = pd.read_csv('wili-2018/labels.csv', sep=';', index_col=0)
    print('Using languages:')
    for index, row in labels.iterrows():
        if index in args.langs:
            print(row['English'])

    trained_model = torch.load(args.modelfile)
    x_test, _, y_test, _ = loadfiles(args.directory)
    dev = torch.device("{}".format("cuda" if torch.cuda.is_available() else "cpu"))

    test_dataset = PrefixLoader(args.langs, x_test, y_test, dev)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    model_eval(test_loader, trained_model, args.eval_mode)
