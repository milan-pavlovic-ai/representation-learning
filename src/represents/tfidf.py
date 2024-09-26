"""TF-IDF Method"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import torch
import numpy as np

from loguru import logger
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import DataManager, TextPreprocessor
from src.model import ModelOptimizer, LinearClassifier, TextRepresentation, TextClassifier


class TfidfDataset(DataManager):
    """Tfidf dataset manager"""

    def __init__(self, sample_size: float = None) -> None:
        """Initializes dataset with texts and labels.

        Args:
            sample_size (float): Take sample from the data. Defaults takes all data.

        Returns:
            None
        """
        super(TfidfDataset, self).__init__(sample_size=sample_size, processor=TextPreprocessor())
        return


class TfidfRepresentation(TextRepresentation):
    """TF-IDF Representation"""

    def __init__(self) -> None:
        """Initializes the TF-IDF vectorizer.

        Returns:
            None
        """
        super(TfidfRepresentation, self).__init__()
        
        self.model = None
        self.prefix = 'tfidf__'

        logger.info('Initialized TF-IDF representation')
        return

    def fit(self, dataset: Any, hparams) -> None:
        """Fit the TF-IDF vectorizer with hyperparameters passed as keyword arguments.
        
        Args:
            dataset: Dataset manager.
            hparams (dict): Hyper-parameters for TfidfVectorizer (e.g., max_features, min_df, etc.).

        Returns:
            None
        """
        # Initialize
        tfidf_hparams = {key.replace(self.prefix, ''): value for key, value in hparams.items() if key.startswith(self.prefix)}
        self.model = TfidfVectorizer(**tfidf_hparams)

        # Train representation on the entire corpus (no batch processing here)
        self.model.fit(dataset.X_train)

        logger.info('TF-IDF Representation has been trained')
        return

    def forward(self, inputs):
        """Transforms input text using the pre-fitted TF-IDF vectorizer.
        
        Args:
            inputs (list of str): List of input text documents to transform.
            
        Returns:
            torch.Tensor: Transformed TF-IDF features in dense array format.
        """
        text_features = self.model.transform(inputs).toarray()
        text_features_tensor = torch.tensor(text_features, dtype=torch.float32)
        return text_features_tensor


class TfidfClassifier(TextClassifier):
    """A classifier that combines a TF-IDF text representation with a linear classifier for binary sentiment classification."""
    
    def __init__(self, hparams: Dict[str, Any]) -> None:
        """Initalize TF-IDF classifier
    
        Args:
            hparams (Any): Hyper-paramters for initialization

        Returns:
            None
        """
        super(TfidfClassifier, self).__init__(hparams=hparams)

        self.representation = TfidfRepresentation()
        self.classifier = LinearClassifier(input_dim=hparams['tfidf__max_features'])

        logger.info('Initialized TF-IDF Classifer')
        return

    def pretrain_representation(self, dataset: Any, hparams) -> None:
        """Pretraining of representation with the TF-IDF vectorizer with hyperparameters passed as keyword arguments.
        
        Args:
            dataset: Dataset manager.
            hparams (dict): Hyper-parameters for TfidfVectorizer (e.g., max_features, min_df, etc.).

        Returns:
            None
        """
        self.representation.fit(dataset=dataset, hparams=hparams)
        return

    def vectorization(self, inputs: Any) -> torch.Tensor:
        """Forward pass through the TF-IDF text representation.
        
        Args:
            inputs (Any): Input text data, usually a list of strings or tokenized data.
            
        Returns:
            torch.Tensor: Vectorized inputs.
        """
        # Convert text to TF-IDF features
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
    dataset = TfidfDataset(sample_size=None)
    dataset.prepare()
    
    # Define hyperparameter space
    param_dists = {
        # TF-IDF Representation Parameters
        'tfidf__max_features': lambda: np.random.randint(1000, 10_000),              # Number of max features
        'tfidf__min_df': lambda: np.random.randint(1, 10),                           # Minimum document frequency
        'tfidf__max_df': lambda: np.random.uniform(0.5, 1),                          # Maximum document frequency 
        'tfidf__ngram_range': lambda: [(1, 1), (1, 2), (1, 3)][np.random.choice(3)], # n-gram for tf-idf representation 
        'tfidf__sublinear_tf': lambda: np.random.choice([True, False]),              # Sublinear term frequency scaling 
        'tfidf__smooth_idf': lambda: np.random.choice([True]),                       # Smooth inverse document frequency 

        # Classifier Hyperparameters
        'clf__learning_rate': lambda: np.random.uniform(1e-5, 1e-2),                # Learning rate 
        'clf__betas': lambda: [(0.9, 0.999), (0.95, 0.999)][np.random.choice(2)],   # Betas for Adam optimizer
        'clf__weight_decay': lambda: np.random.uniform(0, 0.1),                     # Weight decay for regularization
        'clf__amsgrad': lambda: np.random.choice([True, False]),                    # Use AMSGrad variant of Adam optimizer
        'clf__patience': lambda: np.random.randint(10, 30),                         # Early stopping patience
        'clf__num_epochs': lambda: np.random.randint(10, 100)                       # Number of training epochs
    }

    # Run hyperparameter optimization
    optimizer = ModelOptimizer(
        model_class=TfidfClassifier,
        dataset=dataset,
        param_dists=param_dists,
        n_trials=10
    )
    optimizer.random_search()
