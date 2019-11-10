# LT2316-a1

## Languages chosen for training

The following 10 languages were chosen for training and evaluation:

| Language    | Language code |
|-------------|---------------|
| Ukranian    | ukr           |
| Russian     | rus           |
| Bulgarian   | bul           |
| Belorusian  | bel           |
| Polish      | pl            |
| Rusyn       | rue           |
| Swedish     | swe           |
| Norwegian   | nnol          |
| English     | eng           |
| Old English | ang           |

If you wish to train on these languages or evaluate the trained models in this repo, add --langs 'ukr' 'rus' 'bul' 'bel' 'pl' 'rue' 'swe' 'nnol' 'eng' 'ang' when running the script.

## Data transformation

For this particular task, we had the convenience of working with fixed-length sequences (len 100).
The PrefixLoader class handles the loading, padding etc. After converting the vocab to indexes, the function generate_prefixed initiates numpy vectors of a hundred 0's for each prefix and filling them up until the desired length.* 

Once each sentence prefix has been converted to a padded tensor, each sentence instance is of size 100x100. To be able to feed these tensors along with their labels, the original list of sentence labels is transformed to repeat 100 times (i.e. 1<sub>0</sub>, 1<sub>1</sub>, ... , 1<sub>100</sub>). These are then loaded into the model with the help of torch.DataLoader.

* (This is simply because I learned about torch's pad_sequence after I wrote the function and it didn't hamper performance in any significant way. This also means training was done without padding.)

## Command line arguments

### main.py

| Command    | Default | Required? | Description | Example |
|-------------|---------------|---------------|---------------|---------------|
| --directory   | - | Yes | The directory containing the test and training text files | --directory | '/usr/local/courses/lt2316-h19/a1/' |
| --langs   | - | Yes | The list of language codes to train and evaluate on | --langs 'ukr' 'rus' 'bul' 'bel' 'pl' 'rue' 'swe' 'nnol' 'eng' 'ang' |
| --batch_size   | 200             | No | If you wish to train *without* minibatches, set batch_size to 1. | |
| --num_epochs   | 20             | No | - | |

### eval.py

| Command    | Default | Required? | Description | Example |
|-------------|---------------|---------------|---------------|---------------|
| --directory   | - | Yes | The directory containing the test and training text files | --directory '/usr/local/courses/lt2316-h19/a1/' |
| --modelfile | - | Yes | Name of the file containing the trained model | --modefile trained_model_e20b200 |
| --langs   | - | Yes | The list of language codes to train and evaluate on | Same as in main.py. Don't forget to use the same languages for evaluation! |
| --eval_mode   | - | Yes | One of two eval_mode options – see README | --eval_mode 2 |

## Hyperparameters

The batched training data produced fairly low loss values even at a learning rate of 0.01. The single instance training, however, stopped learning at this lr (on a loss of ~28) and was therefore trained at lr=00.01.

There are three different loss modes: mode 1 is "vanilla" NLLL, mode 2 is the former with penalization by 1/100 of the prefix length and mode 3 means penalization by the full prefix length. They are specified like so in the code:

```
if loss_mode == 1:
    loss = criterion(y_pred, labels)
if loss_mode == 2:
    # penalize by 1/100 of the prefix length
    loss = criterion(y_pred, labels) * (len(y_pred.nonzero()))/100 
    # penalize by the prefix length
if loss_mode == 3:
    loss = criterion(y_pred, label) * len(y_pred.nonzero())
```


## Evaluation
The evaluation script uses a non shuffled version of the DataLoader with a batch size of 100 to get the 100 prefixes per sentence.

There are two possible evaluation modes; 1 and 2.

### eval_mode 1:
* At which prefix length the model scored a hit
* The overall accuracy (if the correct class was ever in the top of the predictions)
* The number of characters until hit score for each sentence

### eval_mode 2:
* Total amount of complete failures
* Average prefix length until hit score

The average number of characters until hit score is calculated over the amount of sentences where the model makes a correct prediction, i.e. the total failures are not counted.

*For some reason this script works fine in a Jupyter notebook but gives different outcomes when run from the command line. This type of behaviour suggests that the model is not actually in eval mode while evaluating. *

## Outcome
Minibatching sped up the training process by hours (the training times were not timed, but feel free to try and see ;)). It is possible that this process could be sped up with the use of packing.

### model: trained = trained_batches 20 epochs, loss_mode 1

```
Overall accuracy: 35.0 %

Total amounts of complete failures: 29

Average number of characters until hit score: 8

```

### model: trained_batches 20 epochs, loss_mode 2

```
Overall accuracy: 20.0 %

Total amounts of complete failures: 32

Average number of characters until hit score: 4

```
### model: trained = trained_batches 20 epochs, loss_mode 3
```
Overall accuracy: 40.0 %

Total amounts of complete failures: 24

Average number of characters until hit score: 5
```

