"""T5 (Text-To-Text Transfer Transformer)

    A transformer-based model designed to convert every NLP task into a text-to-text format.
    This means both input and output are in text form, whether it's translation, summarization, or classification.
    It is known for its flexibility in multitask learning.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import torch
import numpy as np

from loguru import logger
from typing import Dict, Any
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.data import DataManager, TextPreprocessor
from src.model import ModelOptimizer, LinearClassifier, TextRepresentation, TextClassifier


class T5Dataset(DataManager):
    """T5 dataset manager"""

    def __init__(self, sample_size: float = None) -> None:
        """Initializes dataset with texts and labels.

        Args:
            sample_size (float): Take sample from the data. Defaults takes all data.

        Returns:
            None
        """
        super(T5Dataset, self).__init__(sample_size=sample_size, processor=None)
        return

    def prepare(self) -> None:
        """Prepare T5 dataset"""
        super().prepare()
        return


class T5Representation(TextRepresentation):
    """T5 Representation"""

    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initializes the T5 vectorizer.

        Args:
            hparams (Dict[str: Any]): Hyper-parameters.
            dataset (Any): Dataset manager.

        Returns:
            None
        """
        super(T5Representation, self).__init__(
            hparams=hparams,
            dataset=dataset
        )
        
        # Hyper-parameters
        self.prefix = 't5__'
        self.t5_hparams = {key.replace(self.prefix, ''): value for key, value in self.hparams.items() if key.startswith(self.prefix)}

        self.use_fine_tuning = self.t5_hparams['fine_tuning']
        self.pretrained_model_name = self.t5_hparams['pretrained_model']

        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_model_name)

        # T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(self.pretrained_model_name)
        self.output_dim = self.model.config.d_model

        # Freeze weights if only use pretrained model
        if not self.use_fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.device = torch.device('cpu')

        logger.info('Initialized T5 representation')
        return

    def vectorize(self, inputs: Any) -> Any:
        """Vectorize words the T5 tokenizer.
        
        Args:
            inputs (Any): Raw inputs. Defualts to None.
        
        Returns:
            Any: Vectorized words
        """
        inputs_vectorized = self.tokenizer(
            inputs,
            return_tensors='pt',                            # PyTorch format
            padding=True,                                   # Pad all inputs to the same length
            truncation=True,                                # Truncate inputs longer than max length
            max_length=self.t5_hparams['sequence_len']      # Maximum sentence length (number of tokens)
        )
        
        return inputs_vectorized

    def forward(self, inputs: Any) -> torch.Tensor:
        """Transforms input vectors using the T5.
        
        Args:
            inputs (Any): Vectors.
            
        Returns:
            torch.Tensor: Transformed T5 features in dense array format.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        t5_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        output = t5_outputs.last_hidden_state[:, 0, :]    # CLS token

        return output


class T5Classifier(TextClassifier):
    """A classifier that combines a T5 text representation with a linear classifier for binary sentiment classification."""
    
    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initalize T5 classifier
    
        Args:
            hparams (Any): Hyper-paramters for initialization.
            dataset: Dataset manager.

        Returns:
            None
        """
        super(T5Classifier, self).__init__(
            hparams=hparams,
            dataset=dataset
        )

        if self.hparams['t5__fine_tuning']:
            self.device = torch.device('cpu')

        self.representation = T5Representation(hparams=self.hparams, dataset=self.dataset)
        self.classifier = LinearClassifier(input_dim=self.representation.output_dim)

        logger.info('Initialized T5 Classifer')
        return

    def prepare(self) -> None:
        """Prepare representation with the T5.

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
        # Convert text to T5 features
        text_features = self.representation(inputs)
        
        # Linear classifier
        outputs_prob = self.classifier(text_features)

        return outputs_prob


if __name__ == "__main__":
   
    # Prepare dataset
    dataset = T5Dataset(sample_size=None)
    dataset.prepare()
    
    # Define hyperparameter space
    param_dists = {
        # T5 Representation Parameters
        't5__sequence_len': lambda: np.random.randint(32, 129),                   # Number of tokens in sentence (document)
        't5__fine_tuning': lambda: np.random.choice([False]),               # Fine tune pretrained models or not
        't5__pretrained_model': lambda: np.random.choice([
            't5-small',
            't5-base',
            # 't5-large',
            # 'google/t5-v1_1-base'
        ]),                                                                         # Pretrained Models

        # Classifier Hyperparameters
        'clf__learning_rate': lambda: 10 ** np.random.uniform(-5, -1),              # Learning rate 
        'clf__betas': lambda: [(0.9, 0.999), (0.95, 0.999)][np.random.choice(2)],   # Betas for Adam optimizer
        'clf__weight_decay': lambda: 10 ** np.random.uniform(-5, -1),               # Weight decay for regularization
        'clf__amsgrad': lambda: np.random.choice([True, False]),                    # Use AMSGrad variant of Adam optimizer
        'clf__patience': lambda: np.random.randint(10, 30),                         # Early stopping patience
        'clf__num_epochs': lambda: np.random.randint(2, 3)                       # Number of training epochs
    }

    # Run hyperparameter optimization
    optimizer = ModelOptimizer(
        model_class=T5Classifier,
        dataset=dataset,
        param_dists=param_dists,
        n_trials=1
    )
    optimizer.random_search()
