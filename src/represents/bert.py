"""BERT (Bidirectional Encoder Representations from Transformers)

    A transformer-based model designed to pretrain deep bidirectional representations by jointly conditioning on both left and right context.
    It is highly effective for many NLP tasks, including question answering, classification, and named entity recognition.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import torch
import numpy as np

from loguru import logger
from typing import Dict, Any
from transformers import BertModel, BertTokenizer, BertConfig

from src.data import DataManager, TextPreprocessor
from src.model import ModelOptimizer, LinearClassifier, TextRepresentation, TextClassifier


class BERTDataset(DataManager):
    """BERT dataset manager"""

    def __init__(self, sample_size: float = None, preprocess: bool = True) -> None:
        """Initializes dataset with texts and labels.

        Args:
            sample_size (float): Take sample from the data. Defaults takes all data. Defaults to None.
            preprocess (bool): Whether to use preprocesed dataset or not. Defaults to True.

        Returns:
            None
        """
        processor = TextPreprocessor() if preprocess else None
        super(BERTDataset, self).__init__(sample_size=sample_size, processor=processor)
        return

    def prepare(self) -> None:
        """Prepare BERT dataset"""
        super().prepare()
        return


class BERTRepresentation(TextRepresentation):
    """BERT Representation"""

    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initializes the BERT vectorizer.

        Args:
            hparams (Dict[str: Any]): Hyper-parameters.
            dataset (Any): Dataset manager.

        Returns:
            None
        """
        super(BERTRepresentation, self).__init__(
            hparams=hparams,
            dataset=dataset
        )
        
        # Hyper-parameters
        self.prefix = 'bert__'
        self.bert_hparams = {key.replace(self.prefix, ''): value for key, value in self.hparams.items() if key.startswith(self.prefix)}

        self.use_fine_tuning = self.bert_hparams['fine_tuning']
        self.pretrained_model_name = self.bert_hparams['pretrained_model']

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)

        # BERT model
        self.model = BertModel.from_pretrained(self.pretrained_model_name)

        # Configuration
        self.config = BertConfig.from_pretrained(self.pretrained_model_name)
        self.output_dim = self.config.hidden_size

        # Freeze weights if only use pretrained model
        if not self.use_fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.device = torch.device('cpu')

        logger.info('Initialized BERT representation')
        return

    def vectorize(self, inputs: Any) -> Any:
        """Vectorize words with BERT tokenizer.

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
            max_length=self.bert_hparams['sequence_len']    # Maximum sentence length (number of tokens)
        )

        return inputs_vectorized

    def forward(self, inputs: Any) -> torch.Tensor:
        """Transforms input vectors using the BERT.
        
        Args:
            inputs (Any): Vectors.
            
        Returns:
            torch.Tensor: Transformed BERT features in dense array format.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        bert_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        output = bert_outputs.last_hidden_state[:, 0, :]    # CLS token

        return output


class BERTClassifier(TextClassifier):
    """A classifier that combines a BERT text representation with a linear classifier for binary sentiment classification."""
    
    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initalize BERT classifier
    
        Args:
            hparams (Any): Hyper-paramters for initialization.
            dataset: Dataset manager.

        Returns:
            None
        """
        super(BERTClassifier, self).__init__(
            hparams=hparams,
            dataset=dataset
        )

        if self.hparams['bert__fine_tuning']:
            self.device = torch.device('cpu')

        self.representation = BERTRepresentation(hparams=self.hparams, dataset=self.dataset)
        self.classifier = LinearClassifier(input_dim=self.representation.output_dim)

        logger.info('Initialized BERT Classifer')
        return

    def prepare(self) -> None:
        """Prepare representation with the BERT.

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
        # Convert text to BERT features
        text_features = self.representation(inputs)

        # Linear classifier
        outputs_prob = self.classifier(text_features)

        return outputs_prob


if __name__ == "__main__":
   
    # Prepare dataset
    dataset = BERTDataset(
        sample_size=None,
        preprocess=False
    )
    dataset.prepare()
    
    # Define hyperparameter space
    param_dists = {
        # BERT Representation Parameters
        'bert__sequence_len': lambda: np.random.randint(32, 129),                   # Number of tokens in sentence (document)
        'bert__fine_tuning': lambda: np.random.choice([True, False]),               # Fine tune pretrained models or not
        'bert__pretrained_model': lambda: np.random.choice([
            'bert-base-uncased',
            # 'bert-large-uncased',
            'bert-base-cased',
            # 'bert-large-cased'
        ]),                                                                         # Pretrained Models

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
        model_class=BERTClassifier,
        dataset=dataset,
        param_dists=param_dists,
        n_trials=10
    )
    optimizer.random_search()
