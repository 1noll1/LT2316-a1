import argparse
import torch
from main import loadfiles
from prefixloader import PrefixLoader
from torch.utils.data import DataLoader
import pandas as pd
#from tqdm import tqdm
import pickle


def model_eval(test_loader, model, train_dataset, eval_mode=1):
    model = model.eval()
    a = 0  # accurate predictions
    predictions = {'Instance {}'.format(i): 0 for i in range(1, len(test_loader) + 1)}
    failures = 0
    scored = 0
    total = 0  # of the instances that are not complete failures
    print('model', model)

    char_indexes = test_dataset.char_index
    index_chars = {idx:char for char, idx in char_indexes.items()}

    class_indexes = train_dataset.class_index
    index_class = {index: char for char, index in class_indexes.items()}

    with torch.no_grad():
        for prefixes, labels in test_loader:
            sent = ''.join([index_chars[idx.item()] for idx in prefixes[0][-1]])
            print('sent:', sent)
            a += 1
            outputs = model(prefixes)
            outputs = outputs.view(1, 100, 100, len(class_indexes))
            _, predicted = torch.max(outputs.data, 3)
            # we're only interested in the output of the last hs per prefix
            predicted = predicted[:, :, -1]
            print('label:', index_class[labels[0][0].item()])

            if eval_mode == 1:

                for i, label in enumerate(labels[0]):
                    if label == predicted[0][i]:
                        predictions['Instance {}'.format(a)] += 1
                        print(f'Correct guess at prefix length {i}')
                        break
                    else:
                        print('Wrong guess:', index_class[predicted[0][i].item()])
                        pass

            elif eval_mode == 2:

                for i, predicted_label in enumerate(predicted[0]):
                    if predicted_label != labels[0][i]:
                        if i == 99:
                            print('Complete failure!')
                            failures += 1
                    elif predicted_label == labels[0][0]:
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

    parser.add_argument("directory", type=str, default='wili-2018/',
                        help="The directory containing the test and training files")
    parser.add_argument("--modelfile", type=str, required=True, help="Name of the PATH to the trained model")
    parser.add_argument('-l', '--langs', nargs='+', help='List of languages to train evaluate on', default=['ukr', 'rus', 'bul', 'bel', 'pol', 'rue', 'swe', 'nno', 'eng', 'ang'])
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
    dev = trained_model.dev

    train_dataset = pickle.load(open("train_dataset.pkl", 'rb'))
    test_dataset = PrefixLoader(args.langs, x_test, y_test, dev, train_dataset)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    model_eval(test_loader, trained_model, train_dataset, args.eval_mode)
