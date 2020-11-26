import numpy as np
import torch


class PrefixLoader():
    def __init__(self, langs, x_file, y_file, dev, train_dataset):
        self.langs = langs
        self.sentences = [(label, sent) for label, sent in zip(y_file, x_file) if label in self.langs]
        self.len = len(self.sentences)
        self.dev = dev

        def get_vocab():
            print('Creating vocab')
            vocab = [' ', 'UNK']
            for label, sent in self.sentences:
                for word in sent.split():
                    vocab += word
            uniquevocab = list(set(vocab))
            uniquevocab.sort()
            vocab_size = len(uniquevocab)
            return vocab, vocab_size, uniquevocab

        if not train_dataset:
            self.vocab, self.vocab_size, self.unique_vocab =\
                get_vocab()
        else:
            self.vocab, self.vocab_size, self.unique_vocab =\
                train_dataset.vocab, train_dataset.vocab_size, sorted(list(set(train_dataset.vocab)))

        def get_vocab_index():
            vocab_index = {}
            for i in range(len(self.unique_vocab)):
                vocab_index.update({self.unique_vocab[i]: i})
            return vocab_index

        self.char_index = get_vocab_index()

        def label_sents():
            print('Converting characters to ints')
            index_sents = []
            for label, sent in self.sentences:
                sent = sent[:100]
                sent = [self.char_index[w] if w in self.vocab else self.char_index['UNK'] for w in sent]
                index_sents.extend([(label, sent)])
            return index_sents

        self.labeled_sentences = label_sents()

        def generate_prefixed():
            print('Padding sentences')
            # pad prefixes with zeros up to length 100
            lang_labels = [lang for lang, sentence in self.labeled_sentences]
            X = []
            for label, sentence in self.labeled_sentences:
                prefix_vectors = []
                seq = np.zeros(100)
                i = 0
                for _int in sentence:
                    seq[i] = _int
                    i += 1
                    prefix_vectors.append(np.copy(seq))
                X.append(prefix_vectors)

            return lang_labels, X

        lang_labels, self.x_train = generate_prefixed()

        def get_class_index():
            class_int = [i for i in range(len(self.langs))]
            return {classname: i for classname, i in zip(self.langs, class_int)}

        self.class_index = get_class_index()
        self.num_classes = len(self.class_index)

        # because we have 100 x 100 sentences, the classes also need to be 100 x 100:
        #self.y_train = [self.class_index[lang] for lang in lang_labels]
        self.y_train = [[self.class_index[lang]] * 100 for lang in lang_labels]

        self.x_tensors = torch.LongTensor(self.x_train)
        self.y_tensors = torch.LongTensor(self.y_train)

        print(self.x_tensors.shape)
        #raise ValueError

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.x_tensors[i].to(self.dev), self.y_tensors[i].to(self.dev)
