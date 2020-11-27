# LT2316-a1

## Languages chosen for training

The following 10 languages were chosen for training and evaluation:

| Language    | Language code |
|-------------|---------------|
| Ukranian    | ukr           |
| Russian     | rus           |
| Bulgarian   | bul           |
| Belorusian  | bel           |
| Polish      | pol           |
| Rusyn       | rue           |
| Swedish     | swe           |
| Norwegian   | nno           |
| English     | eng           |
| Old English | ang           |

If you wish to train on these languages or evaluate the trained models in this repo, add --langs 'ukr' 'rus' 'bul' 'bel' 'pol' 'rue' 'swe' 'nno' 'eng' 'ang' when running the script.

## Data transformation

For this particular task, we had the convenience of working with fixed-length sequences (len 100).
The PrefixLoader class handles the loading, padding etc. After converting the vocab to indexes, the function generate_prefixed initiates numpy vectors of a hundred 0's for each prefix and filling them up until the desired length.* 

Once each sentence prefix has been converted to a padded tensor, each sentence instance is of size 100x100. To be able to feed these tensors along with their labels, the original list of sentence labels is transformed to repeat 100 times (i.e. 1<sub>0</sub>, 1<sub>1</sub>, ... , 1<sub>100</sub>). These are then loaded into the model with the help of torch.DataLoader.

## Command line arguments

### main.py

| Command    | Default | Required? | Description | Example |
|-------------|---------------|---------------|---------------|---------------|
| --directory   | - | Yes | The directory containing the test and training text files | --directory | '/usr/local/courses/lt2316-h19/a1/' |
| --langs   | - | Yes | The list of language codes to train and evaluate on | --langs 'ukr' 'rus' 'bul' 'bel' 'pl' 'rue' 'swe' 'nnol' 'eng' 'ang' |
| --batch_size   | 200             | No | If you wish to train *without* minibatching set batch_size to 1. | |
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

### model: 10 epochs, batch size 16, loss mode 1

```
Overall accuracy: 99.5 %  

Total amounts of complete failures: 25  

Average number of characters until hit score: 6  

```
### model: 10 epochs, batch size 16, loss mode 2

```
Overall accuracy: 99.42 %

Total amounts of complete failures: 29

Average number of characters until hit score: 6  

```

### model: 10 epochs, batch size 16, loss mode 3

```
Overall accuracy: 99.24 %

Total amounts of complete failures: 38

Average number of characters until hit score: 7

```

### model: 10 epochs, batch size 32, loss mode 1
```
Overall accuracy: 99.62 %

Total amounts of complete failures: 19

Average number of characters until hit score: 6
```

### model: 1 epoch, batch size 16, loss mode 1
```
Overall accuracy: 97.54 % %

Total amounts of complete failures: 123

Average number of characters until hit score: 11
```

# Successful examples
These samples are all from the model trained for 10 epochs (loss mode 1).

Sometimes the "reasoning" of the model seems obvious, and sometimes less so.

_"Ēadweard Æþelinges Īegland næs Canadan underrīce æt þæs rīces forma geþoftunge ac weard underrīce on"_  
true label: ang
correct guess at prefix length 0

**Comment:** The character "Ē" is not part of any of the languges alphabets and is thus highly unlikely to appear in any other language than Old English.

_"Gura Pahār är en kulle i Indien. Den ligger i distriktet Tīkamgarh och delstaten Madhya Pradesh, i d"_   
true label: swe  
wrong guess: ang  
wrong guess: nno  
correct guess at prefix length 2  

**Comment:** The character cluster "Gu" as initial characters is common in both Swedish and Norwegian (e.g. the name Gustav), and perhaps even Old English, so although the initial words are in fact not from any of these languages, the model "got lucky".

_"Сюжет е росповѣдь, опис подѣй, епизодох, образох, описох, сцен, котры читательови (слухательови, поз"_  
true label: rue  
wrong guess: rus  
wrong guess: bul  
wrong guess: bul  
correct guess at prefix length 3  

**Comment:** The term Сюжет ("plot") occurs in both Russian and Bulgarian, so it makes sense for the model to make these guesses; how it ended up catching the right label is a mystery. The character ѣ should be the strongest clue, as it is obsolete in both Russian and Bulgarian.

# An example of a complete failure
Here the 'true label' wrong; the language in this fragment is actually German.  
The model does a good job of guessing (it names the Germanic languages).  

_"Walter Benjamin: Briefe an Siegfried Kracauer. Mit 4 Briefen von Siegfried Kracauer an Walter Benjam"_  
*true label: bul  
wrong guess: pol  
wrong guess: ang  
wrong guess: swe  
wrong guess: swe  
wrong guess: eng  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: nno  
wrong guess: swe  
wrong guess: pol  
wrong guess: nno  
wrong guess: swe  
wrong guess: nno  
wrong guess: swe  
wrong guess: pol  
wrong guess: pol  
wrong guess: pol  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: swe  
wrong guess: swe  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: nno  
wrong guess: nno  
wrong guess: ang  
wrong guess: nno  
wrong guess: nno  
wrong guess: ang  
wrong guess: ang  
wrong guess: swe  
wrong guess: swe  
wrong guess: pol  
wrong guess: pol  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: swe  
wrong guess: nno  
wrong guess: swe  
wrong guess: swe  
wrong guess: nno  
wrong guess: ang  
wrong guess: eng  
wrong guess: eng  
wrong guess: eng  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: swe  
wrong guess: nno  
wrong guess: nno  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: swe  
wrong guess: swe  
wrong guess: pol  
wrong guess: nno  
wrong guess: nno  
wrong guess: swe  
wrong guess: nno  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: ang  
wrong guess: swe  
wrong guess: pol  
wrong guess: ang  
wrong guess: swe  
Complete failure!


