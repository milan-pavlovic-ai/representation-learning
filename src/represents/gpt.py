"""GPT (Generative Pretrained Transformer)

    A transformer-based model designed for generating text.
    GPT is pretrained on a large corpus of text in an unsupervised manner and fine-tuned for specific downstream tasks.
    It is widely used for text generation, completion, and other language tasks.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import torch
import numpy as np

from loguru import logger
from typing import Dict, Any
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model

from src.data import DataManager, TextPreprocessor
from src.model import ModelOptimizer, LinearClassifier, TextRepresentation, TextClassifier


class GPTDataset(DataManager):
    """GPT dataset manager"""

    def __init__(self, sample_size: float = None) -> None:
        """Initializes dataset with texts and labels.

        Args:
            sample_size (float): Take sample from the data. Defaults takes all data.

        Returns:
            None
        """
        super(GPTDataset, self).__init__(sample_size=sample_size, processor=None)
        return

    def prepare(self) -> None:
        """Prepare GPT dataset"""
        super().prepare()
        return


class GPTRepresentation(TextRepresentation):
    """GPT Representation"""

    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initializes the GPT vectorizer.

        Args:
            hparams (Dict[str: Any]): Hyper-parameters.
            dataset (Any): Dataset manager.

        Returns:
            None
        """
        super(GPTRepresentation, self).__init__(
            hparams=hparams,
            dataset=dataset
        )
        
        # Hyper-parameters
        self.prefix = 'gpt__'
        self.gpt_hparams = {key.replace(self.prefix, ''): value for key, value in self.hparams.items() if key.startswith(self.prefix)}

        self.use_fine_tuning = self.gpt_hparams['fine_tuning']
        self.pretrained_model_name = self.gpt_hparams['pretrained_model']

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model_name)

        # GPT model
        self.model = GPT2Model.from_pretrained(self.pretrained_model_name)

        # Configuration
        self.config = GPT2Config.from_pretrained(self.pretrained_model_name)
        self.output_dim = self.config.hidden_size

        # Freeze weights if only use pretrained model
        if not self.use_fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.device = torch.device('cpu')

        logger.info('Initialized GPT representation')
        return

    def vectorize(self, inputs: Any) -> Any:
        """Vectorize words the GPT tokenizer.

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
            max_length=self.gpt_hparams['sequence_len']     # Maximum sentence length (number of tokens)
        )

        return inputs_vectorized

    def forward(self, inputs: Any) -> torch.Tensor:
        """Transforms input vectors using the GPT.
        
        Args:
            inputs (Any): Vectors.
            
        Returns:
            torch.Tensor: Transformed GPT features in dense array format.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        gpt_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        output = gpt_outputs.last_hidden_state[:, -1, :]    # CLS token, 0 because it causal and not bi-direct

        return output


class GPTClassifier(TextClassifier):
    """A classifier that combines a GPT text representation with a linear classifier for binary sentiment classification."""
    
    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initalize GPT classifier
    
        Args:
            hparams (Any): Hyper-paramters for initialization.
            dataset: Dataset manager.

        Returns:
            None
        """
        super(GPTClassifier, self).__init__(
            hparams=hparams,
            dataset=dataset
        )

        if self.hparams['gpt__fine_tuning']:
            self.device = torch.device('cpu')

        self.representation = GPTRepresentation(hparams=self.hparams, dataset=self.dataset)
        self.classifier = LinearClassifier(input_dim=self.representation.output_dim)

        logger.info('Initialized GPT Classifer')
        return

    def prepare(self) -> None:
        """Prepare representation with the GPT.

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
        # Convert text to GPT features
        text_features = self.representation(inputs)
        
        # Linear classifier
        outputs_prob = self.classifier(text_features)

        return outputs_prob


if __name__ == "__main__":
   
    # Prepare dataset
    dataset = GPTDataset(sample_size=None)
    dataset.prepare()
    
    # Define hyperparameter space
    param_dists = {
        # GPT Representation Parameters
        'gpt__sequence_len': lambda: np.random.randint(32, 129),                   # Number of tokens in sentence (document)
        'gpt__fine_tuning': lambda: np.random.choice([True, False]),               # Fine tune pretrained models or not
        'gpt__pretrained_model': lambda: np.random.choice([
            'gpt2',
            'gpt2-medium',
            'gpt2-large',
            'distilgpt-2'
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
        model_class=GPTClassifier,
        dataset=dataset,
        param_dists=param_dists,
        n_trials=1
    )
    optimizer.random_search()
