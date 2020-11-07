import numpy as np
import torch


class PrefixLoader():
    def __init__(self, langs, x_file, y_file, dev):
        self.langs = langs
        sentences = [(label, sent) for label, sent in zip(y_file, x_file) if label in self.langs]
        self.len = len(sentences)

        vocab = [' ']
        for x, y in sentences:
            for word in y.split():
                vocab += word
        uniquevocab = list(set(vocab))
        self.vocab_size = len(uniquevocab)

        def get_vocab_index(uniquevocab):
            vocab_index = {}
            for i in range(len(uniquevocab)):
                vocab_index.update({uniquevocab[i]: i})
            return vocab_index

        self.char_index = get_vocab_index(uniquevocab)

        def label_sents(lang_tuple):
            return [(label, [self.char_index[word] for word in sent[:100]]) for label, sent in sentences]

        self.sentences = label_sents(sentences)

        def generate_prefixed(labeled_sentences):
            # pad prefixes with zeros up to length 100
            lang_labels = [lang for lang, sentence in labeled_sentences]
            X = []
            for label, sentence in labeled_sentences:
                prefix_vectors = []
                seq = np.zeros(100)
                i = 0
                for _int in sentence:
                    seq[i] = _int
                    i += 1
                    prefix_vectors.append(np.copy(seq))
                X.append(prefix_vectors)

            return lang_labels, X

        lang_labels, self.x_train = generate_prefixed(self.sentences)

        def class_index(langs):
            classint = [i for i in range(len(self.langs))]
            return {classname: i for classname, i in zip(self.langs, classint)}

        self.class_index = class_index(self.langs)
        self.num_classes = len(self.class_index)

        # because we have 100 x 100 sentences, the classes also need to be 100 x 100:
        self.y_train = [[self.class_index[lang]] * 100 for lang in lang_labels]

        self.x_tensors = torch.LongTensor(self.x_train).to(dev)
        self.y_tensors = torch.LongTensor(self.y_train).to(dev)

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.x_tensors[i], self.y_tensors[i]
