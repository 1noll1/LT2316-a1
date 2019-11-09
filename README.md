# LT2316-a1

## Languages chosen for training

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

## Data transformation

For this particular task, we had the convenience of working with fixed-length sequences (len 100).
The PrefixLoader class handles the loading, padding etc. After converting the vocab to indexes, the function generate_prefixed initiates numpy vectors of a hundred 0's for each prefix and filling them up until the desired length.* 

Once each sentence prefix has been converted to a padded tensor, each sentence instance is of size 100x100. To be able to feed these tensors along with their labels, the original list of sentence labels is transformed to repeat 100 times (i.e. 1<sub>0</sub>, 1<sub>1</sub>, ... , 1<sub>100</sub>).

* (This is simply because I learned about torch's pad_sequence after I wrote the function and it didn't hamper performance in any significant way.)

## Command line arguments


| Command    | Default | Required? | Description |
|-------------|---------------|---------------|---------------|
| --directory   | - | Yes | The directory containing the test and training text files |
| --batch_size   | 200             | No | If you wish to train *without* minibatches, set batch_size to 1. |
| --num_epochs   | 20             | No | - |


## Hyperparameters

The batched training data produced fairly low loss values even at a learning rate of 0.1. The single instance training, however, stopped learning at this lr and was therefore trained at lr=0.01.

## Evaluation
The evaluation script uses a non shuffled version of the DataLoader with a batch size of 100 to get the 100 prefixes per sentence.
