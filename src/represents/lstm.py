"""LSTM (Long Short-Term Memory)

    A type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data, such as time series or natural language.
    LSTM cells address the vanishing gradient problem common in traditional RNNs, allowing them to better remember long-term patterns in data.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import torch
import torch.nn as nn
import numpy as np

from loguru import logger
from typing import Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from torch.nn.utils.rnn import pad_sequence

from src.data import DataManager, TextPreprocessor
from src.model import ModelOptimizer, LinearClassifier, TextRepresentation, TextClassifier


class LSTMDataset(DataManager):
    """LSTM dataset manager"""

    def __init__(self, sample_size: float = None) -> None:
        """Initializes dataset with texts and labels.

        Args:
            sample_size (float): Take sample from the data. Defaults takes all data.

        Returns:
            None
        """
        super(LSTMDataset, self).__init__(sample_size=sample_size, processor=TextPreprocessor())
        return

    def prepare(self) -> None:
        """Prepare LSTM dataset"""
        super().prepare()
        return


class LSTMRepresentation(TextRepresentation):
    """LSTM Representation"""

    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initializes the LSTM vectorizer.

        Args:
            hparams (Dict[str, Any]): Hyper-parameters.
            dataset (Any): Dataset manager.

        Returns:
            None
        """
        super(LSTMRepresentation, self).__init__(
            hparams=hparams,
            dataset=dataset
        )
        
        # Hyper-paramters
        self.prefix = 'lstm__'
        self.lstm_hparams = {key.replace(self.prefix, ''): value for key, value in self.hparams.items() if key.startswith(self.prefix)}

        self.embedding_dim = int(self.lstm_hparams['embedding_dim'])
        self.hidden_dim = int(self.lstm_hparams['hidden_dim'])
        self.num_layers = int(self.lstm_hparams['n_layers'])
        self.is_bidirect = bool(self.lstm_hparams['bidirectional'])
        self.dropout_prob = float(self.lstm_hparams['dropout'])

        # Vectorizer
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.dataset.X_train)
        self.vocab = self.vectorizer.vocabulary_
        self.analyzer = self.vectorizer.build_analyzer()

        # Embedding
        num_embeddings = len(self.vocab) + 2
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=self.embedding_dim 
        )

        # LSTM model
        self.model = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.is_bidirect,
            dropout=self.dropout_prob,
            batch_first=True
        )

        self.output_dim = 2 * self.hidden_dim if self.is_bidirect else self.hidden_dim

        logger.info('Initialized LSTM representation')
        return

    def vectorize(self, inputs: Any) -> Any:
        """Create word vectors for embedding layer

        Args:
            inputs (Any): Raw inputs. Defualts to None.

        Returns:
            Any: Vectorized words
        """
        # Vectorizer
        all_docs_tensors = []
        max_sequence_len = self.hparams['vec__sequence_len']

        for doc in inputs:
            tokens = self.analyzer(doc)
            doc_vector = [1 + self.vocab[word] for word in tokens if word in self.vocab]
            doc_tensor = torch.tensor(doc_vector[:max_sequence_len], dtype=torch.long).to(self.device)
            all_docs_tensors.append(doc_tensor)

        all_docs_tensors_final = pad_sequence(all_docs_tensors, batch_first=True, padding_value=0).to(self.device)

        return all_docs_tensors_final

    def forward(self, inputs: Any) -> torch.Tensor:
        """Transforms input vectors using the LSTM.
        
        Args:
            inputs (Any): Vectors.
            
        Returns:
            torch.Tensor: Transformed LSTM features in dense array format.
        """
        # Embedding
        embedded = self.embedding(inputs)  

        # LSTM
        batch_size = embedded.size(0)
        num_directions = 2 if self.is_bidirect else 1

        hidden = (
            torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_dim).to(self.device)
        )

        lstm_out, (hidden, cell) = self.model(embedded, hidden)                     # lstm_out: [batch_size, sentence_length, hidden_dim]

        # For bidirectional concatenate the final forward and backward hidden states
        if self.model.bidirectional:
            output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)     # [batch_size, hidden_dim * 2]
        else:
            output = hidden[-1,:,:]                                         # [batch_size, hidden_dim]

        return output


class LSTMClassifier(TextClassifier):
    """A classifier that combines a LSTM text representation with a linear classifier for binary sentiment classification."""
    
    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initalize LSTM classifier
    
        Args:
            hparams (Any): Hyper-paramters for initialization.
            dataset: Dataset manager.

        Returns:
            None
        """
        super(LSTMClassifier, self).__init__(
            hparams=hparams,
            dataset=dataset
        )

        self.representation = LSTMRepresentation(hparams=self.hparams, dataset=self.dataset)
        self.classifier = LinearClassifier(input_dim=self.representation.output_dim)

        logger.info('Initialized LSTM Classifer')
        return

    def prepare(self) -> None:
        """Prepare representation with the LSTM.

        Returns:
            None
        """
        return

    def encode(self, inputs: Any) -> Any:
        """Encode text representation for embedding layer.
        
        Args:
            inputs (Any): Input text data, usually a list of strings or tokenized data.
            
        Returns:
            Any: Encoder output.
        """
        inputs_vectorized = self.representation.vectorize(inputs)

        return inputs_vectorized

    def forward(self, inputs: Any) -> torch.Tensor:
        """Forward pass through text representation and linear classifier.
        
        Args:
            inputs (Any): Input vectorized data. This is important because of the GPU usage.
            
        Returns:
            torch.Tensor: The sigmoid-activated logits representing the class probabilities.
        """
        # Convert text to LSTM features
        text_features = self.representation(inputs)
        
        # Linear classifier
        outputs_prob = self.classifier(text_features)

        return outputs_prob


if __name__ == "__main__":
   
    # Prepare dataset
    dataset = LSTMDataset(sample_size=None)
    dataset.prepare()
    
    # Define hyperparameter space
    param_dists = {
        # Word vectorizer 
        'vec__sequence_len': lambda: np.random.randint(32, 129),                    # Sequence length during the initial vectorization of words

        # LSTM Representation Parameters
        'lstm__embedding_dim': lambda: np.random.randint(128, 2049),                # Dimensionality of the embedding vectors (size of each word vector)
        'lstm__hidden_dim': lambda: np.random.randint(64, 512),                     # Number of hidden units in each LSTM layer
        'lstm__bidirectional': lambda: np.random.choice([True, False]),             # Whether to use a bidirectional LSTM
        'lstm__n_layers': lambda: np.random.randint(1, 3),                          # Number of LSTM layers (stacked LSTMs)
        'lstm__dropout': lambda: np.random.uniform(0.0, 0.5),                       # Dropout probability for regularization (applied between LSTM layers)

        # Classifier Hyperparameters
        'clf__learning_rate': lambda: 10 ** np.random.uniform(-5, -1),              # Learning rate 
        'clf__betas': lambda: [(0.9, 0.999), (0.95, 0.999)][np.random.choice(2)],   # Betas for Adam optimizer
        'clf__weight_decay': lambda: 10 ** np.random.uniform(-5, -1),               # Weight decay for regularization
        'clf__amsgrad': lambda: np.random.choice([True, False]),                    # Use AMSGrad variant of Adam optimizer
        'clf__patience': lambda: np.random.randint(10, 30),                         # Early stopping patience
        'clf__num_epochs': lambda: np.random.randint(30, 100)                       # Number of training epochs
    }

    # Run hyperparameter optimization
    optimizer = ModelOptimizer(
        model_class=LSTMClassifier,
        dataset=dataset,
        param_dists=param_dists,
        n_trials=10
    )
    optimizer.random_search()
