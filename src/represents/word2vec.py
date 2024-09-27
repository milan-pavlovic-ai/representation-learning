"""Word2Vec Method"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import torch
import torch.nn.functional as F
import numpy as np

from loguru import logger
from typing import Dict, Any
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence

from src.data import DataManager, TextPreprocessor
from src.model import ModelOptimizer, LinearClassifier, TextRepresentation, TextClassifier


class Word2VecDataset(DataManager):
    """Word2Vec dataset manager"""

    def __init__(self, sample_size: float = None) -> None:
        """Initializes dataset with texts and labels.

        Args:
            sample_size (float): Take sample from the data. Defaults takes all data.

        Returns:
            None
        """
        super(Word2VecDataset, self).__init__(sample_size=sample_size, processor=TextPreprocessor())
        return

    def prepare(self) -> None:
        """Prepare Word2Vec dataset"""
        super().prepare()
        # self.into_tokens_lists()
        return


class Word2VecRepresentation(TextRepresentation):
    """Word2Vec Representation"""

    def __init__(self, hparams: Dict[str, Any]) -> None:
        """Initializes the Word2Vec vectorizer.

        Args:
            hparams (Dict[str, Any]): Hyper-parameters.

        Returns:
            None
        """
        super(Word2VecRepresentation, self).__init__(hparams=hparams)
        
        self.prefix_wc = 'w2v__'
        self.word2vec_hparams = {key.replace(self.prefix_wc, ''): value for key, value in self.hparams.items() if key.startswith(self.prefix_wc)}

        self.prefix_enc = 'enc__'
        self.encoder_hparams = {key.replace(self.prefix_enc, ''): value for key, value in self.hparams.items() if key.startswith(self.prefix_enc)}

        self.model = None
        
        self.use_attention = self.encoder_hparams['use_attention']
        self.output_dim = self.word2vec_hparams['vector_size']

        logger.info('Initialized Word2Vec representation')
        return

    def prepare(self, dataset: Any) -> None:
        """Fit the Word2Vec vectorizer with hyperparameters passed as keyword arguments.
        
        Args:
            dataset: Dataset manager.

        Returns:
            None
        """
        # Prepare data
        sentences = dataset.X_train.apply(lambda x: x.split())
        
        # Fit the model
        self.model = Word2Vec(sentences=sentences, **self.word2vec_hparams)

        logger.info('Word2Vec Representation has been trained')
        return

    def forward(self, inputs):
        """Transforms input text using the pre-fitted Word2Vec vectorizer.
        
        Args:
            inputs (list of str): List of input text documents to transform.
            
        Returns:
            torch.Tensor: Transformed Word2Vec features in dense array format.
        """
        # Create embeddings
        embeddings = []
        for doc in inputs:
            words = doc.split()
            doc_embeddings = [self.model.wv[word] for word in words if word in self.model.wv]
            embeddings.append(doc_embeddings)

        # If there are no valid words in any document
        if not any(embeddings):
            return torch.zeros((len(inputs), self.model.vector_size), dtype=torch.float32)

        # Pad sentences to equal size
        embeddings_array = [np.array(doc, dtype=np.float32) for doc in embeddings]
        embeddings_padded = pad_sequence([torch.tensor(doc) for doc in embeddings_array], batch_first=True)
   
        # Create a one embedding per documment based on words embeddings
        if self.use_attention:
            attention_weights = F.softmax(embeddings_padded, dim=1)         # Shape: [batch_size, seq_length, 1]
            weighted_embeddings = embeddings_padded * attention_weights     # Broadcasting to get weighted embeddings
            embeddings_doc = weighted_embeddings.sum(dim=1)                 # Shape: [batch_size, embedding_dim]
        else:
            embeddings_doc = embeddings_padded.mean(dim=1)

        return embeddings_doc


class Word2VecClassifier(TextClassifier):
    """A classifier that combines a Word2Vec text representation with a linear classifier for binary sentiment classification."""
    
    def __init__(self, hparams: Dict[str, Any]) -> None:
        """Initalize Word2Vec classifier
    
        Args:
            hparams (Any): Hyper-paramters for initialization

        Returns:
            None
        """
        super(Word2VecClassifier, self).__init__(hparams=hparams)
        
        self.representation = Word2VecRepresentation(hparams=self.hparams)
        self.classifier = LinearClassifier(input_dim=self.representation.output_dim)

        logger.info('Initialized Word2Vec Classifer')
        return

    def prepare_representation(self, dataset: Any) -> None:
        """Pretraining of representation with the Word2Vec vectorizer with hyperparameters passed as keyword arguments.
        
        Args:
            dataset: Dataset manager.

        Returns:
            None
        """
        self.representation.prepare(dataset=dataset)
        self.representation.to(self.device)
        return

    def encode(self, inputs: Any) -> torch.Tensor:
        """Forward pass through the Word2Vec text representation.
        
        Args:
            inputs (Any): Input text data, usually a list of strings or tokenized data.
            
        Returns:
            torch.Tensor: Encoder output.
        """
        # Convert text to Word2Vec features
        text_features = self.representation(inputs)
        
        return text_features

    def forward(self, inputs: Any) -> torch.Tensor:
        """Forward pass through the linear classifier.
        
        Args:
            inputs (Any): Input vectorized data. This is important because of the GPU usage.
            
        Returns:
            torch.Tensor: The sigmoid-activated logits representing the class probabilities.
        """
        # Linear classifier
        outputs_prob = self.classifier(inputs)

        return outputs_prob


if __name__ == "__main__":
   
    # Prepare dataset
    dataset = Word2VecDataset(sample_size=None)
    dataset.prepare()
    
    # Define hyperparameter space
    param_dists = {
        # Word2Vec Representation Parameters
        'w2v__vector_size': lambda: np.random.randint(128, 2049),                # Dimensionality of the word vectors
        'w2v__window': lambda: np.random.randint(1, 11),                         # Maximum distance between the current and predicted word
        'w2v__min_count': lambda: np.random.randint(1, 10),                      # Minimum frequency count of words to be considered for training
        'w2v__sg': lambda: np.random.randint(0, 2),                              # 0 uses the CBOW approach, 1 uses the Skip-gram approach
        'w2v__alpha': lambda: 10 ** np.random.uniform(-5, -1),                   # The initial learning rate.
        'w2v__negative': lambda: np.random.randint(5, 20),                       # The number of negative samples to use 
        'w2v__epochs': lambda: np.random.randint(5, 20),                         # The number of iterations over the corpus during training
        'w2v__workers': lambda: np.random.randint(10, 12),                       # The number of worker threads to train the model

        # Encoder
        'enc__use_attention': lambda: np.random.choice([False, True]),           # Use attention pooling for document embeddings, otherwise use average

        # Classifier Hyperparameters
        'clf__learning_rate': lambda: 10 ** np.random.uniform(-5, -1),              # Learning rate 
        'clf__betas': lambda: [(0.9, 0.999), (0.95, 0.999)][np.random.choice(2)],   # Betas for Adam optimizer
        'clf__weight_decay': lambda: 10 ** np.random.uniform(-5, -1),               # Weight decay for regularization
        'clf__amsgrad': lambda: np.random.choice([True, False]),                    # Use AMSGrad variant of Adam optimizer
        'clf__patience': lambda: np.random.randint(10, 30),                         # Early stopping patience
        'clf__num_epochs': lambda: np.random.randint(10, 100)                       # Number of training epochs
    }

    # Run hyperparameter optimization
    optimizer = ModelOptimizer(
        model_class=Word2VecClassifier,
        dataset=dataset,
        param_dists=param_dists,
        n_trials=10
    )
    optimizer.random_search()
