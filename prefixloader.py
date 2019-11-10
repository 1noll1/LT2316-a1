from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class PrefixLoader():
    def __init__(self, langs, x_file, y_file, dev):
        self.langs = langs
        sentences = [(label, sent) for label, sent in zip(y_file, x_file) if label in self.langs]
        self.len = len(sentences)
        
        vocab = [' ']
        for x,y in sentences:
            for w in y.split():
                vocab += w
        uniquevocab = list(set(vocab))
        self.vocab_size = len(uniquevocab)
        
        def getvocabindex(uniquevocab):
            vocabindex = {}
            for i in range(len(uniquevocab)):
                vocabindex.update({uniquevocab[i]: i})
            return vocabindex

        self.char_index = getvocabindex(uniquevocab)
        
        def labelsents(lang_tuple):
            return [(label, [[self.char_index[x]] for x in sent[:100]]) for label, sent in sentences]       
              
        self.sentences = labelsents(sentences)
        
        def generate_prefixed(labeled_sentences):
            # pad prefixes with zeros up to length 100
            lang_labels = [lang for lang, sentence in labeled_sentences]
            prefix_vectors = []
            for label, sentence in labeled_sentences:
                seq = np.zeros(100)
                i = 0
                for _int in sentence:
                    seq[i] = _int[0]
                    i += 1
                    prefix_vectors.append(np.copy(seq))
            return lang_labels, prefix_vectors

        lang_labels, self.x_train = generate_prefixed(self.sentences)        
        
        def class_index(langs):
            classint = [i for i in range(len(self.langs))]
            return {classname: i for classname, i in zip(self.langs, classint)}  
        
        self.class_index = class_index(self.langs)        
        self.num_classes = len(self.class_index)
        
        # because we have 100 x 100 sentences, the classes also need to be 100 x 100:
        repeat_y = [np.repeat(self.class_index[lang], 100) for lang in lang_labels]
        flatten = [item for sublist in repeat_y for item in sublist]
        self.y_train = flatten

        self.pairs = list(zip(self.x_train, self.y_train))
        
        self.x_tensors = torch.LongTensor(self.x_train).to(dev)
        self.y_tensors = torch.LongTensor(self.y_train).to(dev)

        print(self.x_tensors[0])
        print(self.y_tensors[0])
        
    def __len__(self):
        return self.len
            
    def __getitem__(self, i):
        return self.x_tensors[i], self.y_tensors[i]
