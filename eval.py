import argparse
import torch
from prefixloader import PrefixLoader
from mainline import loadfiles
from torch.utils.data import Dataset, DataLoader 

def model_eval(test_loader, model, eval_mode=1):
    model.eval()
    a = 0
    predictions = {'Instance {}'.format(i) : 0 for i in range(1, len(test_loader)+1)}
    failures = 0
    scored = 0
    total = 0 # of the instances that are not complete failures
    
    with torch.no_grad():
        for x, y in test_loader:
            a += 1 #a as in accurate!
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            if eval_mode==1:
                for i, value in enumerate(y):
                    if value == predicted[i]:
                        predictions['Instance {}'.format(a)] += 1 
                        print('Correct guess at prefix length {}'.format(i))
                    else:
                        continue
            elif eval_mode==2:
                for i, value in enumerate(predicted): 
                    if value != y[i]: 
                        if i == len(predicted)-1:
                            print('Completele failure!')
                            failures += 1
                        else:
                            pass
                    if value == y[i]:
                        print('Number of characters until hit score: {}'.format(i+1))
                        scored += (i+1)
                        total += 1
                        break
                    
    if eval_mode==1:                
        #print('Number of accurate guesses per instance:')
        for key in predictions:
            print(key,':', predictions[key])

        accurate = 0
        
        print('Overall accuracy:')
        for key in predictions:
            if predictions[key] > 0:
                accurate += 1
        print(accurate/len(predictions) * 100, '%')
        
    if eval_mode==2: 
        print('Total amounts of complete failures: {}'.format(failures))       
        print('Average number of characters until hit score: {}'.format(round(scored/total)))          

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate GRU model.")

    parser.add_argument("--directory", type=str, required=True, help="The directory containing the test and training files")
    parser.add_argument("--modelfile", type=str, required=True, help="Name of the file containing the trained model")
    parser.add_argument('--langs', nargs='+', required=True, default=[], help="The list of language codes to train on")
    parser.add_argument('--eval_mode', type=int, required=True, help="Which of the 2 eval modes to implement – see README")

    args = parser.parse_args()

    trained_model = torch.load(args.modelfile)
    x_test, _, y_test, _ = loadfiles(args.directory)
    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")
    
    dataset = PrefixLoader(args.langs, x_test, y_test, dev)
    test_loader = DataLoader(dataset=dataset, batch_size=100, shuffle=False, num_workers=0)
    model_eval(test_loader, trained_model, args.eval_mode)