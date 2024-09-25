from gensim.models import Word2Vec
import numpy as np

class Word2VecBase:
    def __init__(self, vector_size=100, window=5, min_count=1, sg=0):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.model = None

class Word2VecPretrained(Word2VecBase):
    def __init__(self):
        super().__init__()

    def train(self, dataset):
        sentences = [doc.split() for doc in dataset]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, sg=self.sg)

    def evaluate(self, dataset):
        # Evaluation logic for Word2Vec pretrained
        return self.eval_metric(dataset)

class Word2VecFineTuned(Word2VecBase):
    def __init__(self):
        super().__init__()

    def train(self, dataset):
        sentences = [doc.split() for doc in dataset]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, sg=self.sg)
        # Fine-tune the model on new data
        self.model.train(sentences, total_examples=len(sentences), epochs=5)

    def evaluate(self, dataset):
        # Evaluation logic for fine-tuned Word2Vec
        return self.eval_metric(dataset)

class Word2VecFromScratch(Word2VecBase):
    def __init__(self, vector_size=100, window=5, min_count=1, sg=0):
        super().__init__(vector_size, window, min_count, sg)

    def train(self, dataset):
        sentences = [doc.split() for doc in dataset]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, sg=self.sg)

    def evaluate(self, dataset):
        # Evaluation logic for Word2Vec trained from scratch
        return self.eval_metric(dataset)
