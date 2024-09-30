"""Start application"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from loguru import logger
from typing import Dict, Any
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


class TextRepresentation(nn.Module):
    """Base class for text representation modules.
        - This class serves as a template for implementing various types of text representations.
        - Subclasses should implement the `forward` method to transform text data into a numerical representation.
    """

    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initializes the TextRepresentation module
        
        Args:
            hparams (Dict[str, Any]): Hyper-paramters as dictionary.
            dataset: Dataset manager.

        Returns:
            None
        """
        super(TextRepresentation, self).__init__()
        self.hparams = hparams
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return

    def vectorize(self, inputs: Any) -> Any:
        """Prepare of representation with the vectorizer.

        Args:
            inputs (Any): Inputs (raw words, strings).

        Raises:
            NotImplementedError: This method must be implemented by subclasses of TextRepresentation.

        Returns:
            Any: Vectorized inputs
        """
        raise NotImplementedError("Subclasses should implement this!")

    def forward(self, inputs: Any) -> torch.Tensor:
        """Transforms the input data into a tensor representation.

        Args:
            inputs (Any): The input text data. The specific format of `inputs` will depend on the subclass implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses of TextRepresentation.

        Returns:
            torch.Tensor: The transformed tensor representation of the input text data.
        """
        raise NotImplementedError("Subclasses should implement this!")


class TextClassifier(nn.Module):
    """Text classifier that combines a text representation with a linear classifier for binary sentiment classification."""
    
    def __init__(self, hparams: Dict[str, Any], dataset: Any) -> None:
        """Initalize complete classifier
    
        Args:
            hparams (Dict[str, Any]): Hyper-paramters for initialization
            dataset: Dataset manager.

        Returns:
            None
        """
        super(TextClassifier, self).__init__()
        self.hparams = hparams
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return

    def prepare(self) -> None:
        """Prepare the text representation.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses should implement this!")

    def encode(self, inputs: Any) -> Any:
        """Encode text representation into vectors
        
        Args:
            inputs (Any): Input text data, usually a list of strings or tokenized data.
            
        Returns:
            Any: Encoder vectors.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def forward(self, inputs: Any) -> torch.Tensor:
        """Forward pass through the text representation and main classifier.
        
        Args:
            inputs (Any): Input vectorized data. This is important because of the GPU usage.
            
        Returns:
            torch.Tensor: The sigmoid-activated logits representing the class probabilities.
        """
        raise NotImplementedError("Subclasses should implement this!")


class LinearClassifier(nn.Module):
    """A simple feedforward neural network model for binary sentiment classification"""

    def __init__(self, input_dim: int) -> None:
        """Initializes with a fully connected layer.

        Args:
            input_dim (int): The number of input features.

        Returns:
            None
        """
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        logger.info('Initialized Linear classifier')
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) after applying the sigmoid activation.
        """
        outputs = self.fc(inputs)
        outputs_prob = torch.sigmoid(outputs)
        
        return outputs_prob


class ModelManager:
    """Model Manager"""

    class Config:
        """Model Configuration"""
        THRESHOLD = 0.5

    def __init__(self, model: Any, criterion: nn.Module, optimizer: optim.Optimizer, dataset: Any) -> None:
        """Initializes the ModelTrainer

        Args:
            model (Any): The model to train.
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimizer for model training.
            dataset (ReviewDataset): Processed dataset.

        Returns:
            None
        """
        # Initialize
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset

        # Set device
        self.model.to(self.model.device)
        return

    def train(self, trial_id: int, total_trials: int, num_epochs: int, patience: int) -> None:
        """Trains the model for a specified number of epochs.

        Args:
            trial_id (int): Trial identifier
            total_trials (int): Total number of trials.
            num_epochs (int): The number of epochs to train the model.
            patience (int): Number of epochs to tolerate bad performance.

        Returns:
            None
        """
        logger.info(f'\nStarted training for trial {trial_id}/{total_trials} ...')

        best_f1 = 0
        patience_counter = 0
        version = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        for epoch in tqdm(range(num_epochs), desc='\nTraining'):

            # Training
            torch.cuda.empty_cache()      
            self.model.train()
            
            y_true = []
            y_pred = []
            epoch_loss = 0.0

            for inputs, labels in tqdm(self.dataset.dataloader_train, desc=f'\nTraining epoch {epoch}/{num_epochs}'):

                # Encoder
                inputs = self.model.encode(inputs)

                # Move to GPU
                inputs = inputs.to(self.model.device)
                labels = labels.float().view(-1, 1).to(self.model.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Loss and Backpropagation
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Metrics
                epoch_loss += loss.item()
                predictions = (outputs > ModelManager.Config.THRESHOLD)
                
                y_true.extend(labels.cpu().numpy().astype(int))
                y_pred.extend(predictions.cpu().numpy().astype(int))

            accuracy = accuracy_score(y_true, y_pred) * 100
            f1_result = f1_score(y_true, y_pred, average='binary')
            avg_loss = epoch_loss / len(self.dataset.dataloader_train)
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1_result:.4f}')

            # Validation
            f1_score_val = self.validate()

            # Early stopping
            if f1_score_val > best_f1:
                best_f1 = f1_score_val
                patience_counter = 0
                logger.info(f'Best F1 score has been achieved: {best_f1:.4f}')
                ModelManager.save(self.model, version=version, suffix=str(trial_id))
            else:
                patience_counter += 1
                logger.info(f'No improvement in F1 score. Patience counter: {patience_counter}/{patience}')
                
            if patience_counter >= patience:
                logger.info('Early stopping triggered.')
                break

            torch.cuda.synchronize()

        logger.info('\nTraining has been completed')
        return

    def validate(self) -> float:
        """
        Validates the model on the validation dataset and computes F1 score.

        Returns:
            float: The F1 score of the model on the validation set.
        """
        logger.info('Started validation ...')
        
        self.model.eval()
        y_true = []
        y_pred = []
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataset.dataloader_valid, desc='Validating model'):
                
                # Encoder
                inputs = self.model.encode(inputs)

                # Move to GPU
                torch.cuda.empty_cache()
                inputs = inputs.to(self.model.device)
                labels = labels.float().view(-1, 1).to(self.model.device)
                
                # Predictions
                outputs = self.model(inputs)
                predictions = (outputs > ModelManager.Config.THRESHOLD)
                
                # Metrics
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                y_true.extend(labels.cpu().numpy().astype(int))
                y_pred.extend(predictions.cpu().numpy().astype(int))
                
                torch.cuda.synchronize()

        accuracy = accuracy_score(y_true, y_pred) * 100
        f1_result = f1_score(y_true, y_pred, average='binary')
        average_loss = total_loss / len(self.dataset.dataloader_valid)

        logger.info(f'Validation Avg Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1_result:.4f}')
        logger.info('Validation has been completed.')

        return f1_result

    def test(self, the_model: Any, decimals: int = 4) -> None:
        """Evaluates the model using the test set.

        Args:
            the_model: Model to test.
            decimals (int): Number of decimal places for metric values. Defaults to 4.

        Returns:
            None
        """
        logger.info('\nTesting model ...')

        all_dataloaders = [
            ('Training set', self.dataset.dataloader_train),
            ('Validation set', self.dataset.dataloader_valid),
            ('Testing set', self.dataset.dataloader_test)
        ]

        for dataset_name, dataloader in all_dataloaders:
            all_preds = []
            all_labels = []

            # Calculate predictions
            with torch.no_grad():
                for inputs, labels in tqdm(dataloader, desc=f'Evaluation on {dataset_name}'):
                    # Encoder
                    inputs = the_model.encode(inputs)

                    # Move to GPU
                    inputs = inputs.to(self.model.device)
                    labels = labels.float().view(-1, 1) 
                    
                    # Predictions
                    outputs = the_model(inputs)
                    predictions = (outputs > ModelManager.Config.THRESHOLD)
                    
                    # Metrics
                    all_preds.extend(predictions.cpu().numpy().astype(int))
                    all_labels.extend(labels.cpu().numpy().astype(int))

            # Calculate metrics
            f1_result = round(f1_score(all_labels, all_preds), decimals)
            accuracy = round(accuracy_score(all_labels, all_preds), decimals)
            precision = round(precision_score(all_labels, all_preds), decimals)
            recall = round(recall_score(all_labels, all_preds), decimals)
            auc_score = round(roc_auc_score(all_labels, all_preds), decimals)

            results = {
                'Data Type': dataset_name,
                'Data Size': len(dataloader.dataset),
                'F1 Score': f1_result,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'AUC Score': auc_score
            }

            for metric, value in results.items():
                logger.info(f'{metric}: {value}')

        logger.info('Model has been tested.\n')
        return

    @staticmethod
    def save(the_model: Any, version: str = None, suffix: str = 'tmp') -> None:
        """Save model at given path

        Args:
            the_model: Model to save.
            version: Model version.
            suffix: Suffix for path name. Deafults to empty string.

        Returns:
            None
        """
        if version is None:
            version = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        
        model_name = the_model.__class__.__name__
        model_dir = f'models/{model_name}/'
        model_path = f'{model_dir}{version}_{suffix}_model.pth'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        torch.save(the_model.state_dict(), model_path)
        logger.info(f'Saved model at: {model_path}')
        return


class ModelOptimizer:
    """Optimizing model with hyperparameters using random search"""

    def __init__(self, model_class: Any, dataset: Any, param_dists: Dict[str, Any], n_trials: int) -> None:
        """Initializes with parameter distributions, number of trials, and dataset.

        Args:
            model_class: The class of the combined model with text representation.
            dataset: The dataset object containing the training and validation data.
            param_dists (Dict[str, Any]): A dictionary of hyperparameter distributions.
            n_trials (int): The number of random trials to perform.

        Returns:
            None
        """
        self.model_class = model_class
        self.dataset = dataset
        self.param_dists = param_dists
        self.n_trials = n_trials
        return
    
    def random_search(self) -> None:
        """Performs random search to optimize hyperparameters.

        The method samples hyperparameters from the specified distributions.
        It keeps track of the best performing hyperparameters based on accuracy.

        Returns:
            None
        """
        # Guarantees random start
        np.random.seed(None)
        random.seed(None)

        # Initialize
        best_f1_score: float = 0.0
        best_params: Dict[str, Any] = {}
        best_trial = -1
        best_model = None
        
        for trial_id in range(self.n_trials):
            logger.warning(f'Executing trial {trial_id}')

            # Clean cache
            torch.cuda.empty_cache()

            # Sample hyperparameters
            hparams = {key: distribution() for key, distribution in self.param_dists.items()}
            logger.info("Sampled Hyper-parameters:")
            for key, value in hparams.items():
                logger.info(f"\t{key}: {value}")

            # Create model
            model = self.model_class(hparams, self.dataset)
            model.prepare()

            # Train the model
            criterion = nn.BCELoss(reduction='mean')
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=hparams['clf__learning_rate'],
                betas=hparams['clf__betas'],
                weight_decay=hparams['clf__weight_decay'],
                amsgrad=hparams['clf__amsgrad']
            )

            model_manager = ModelManager(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                dataset=self.dataset
            )
            
            model_manager.train(
                trial_id=trial_id,
                total_trials=self.n_trials,
                num_epochs=hparams['clf__num_epochs'],
                patience=hparams['clf__patience']
            )

            # Validate the model
            f1_result = model_manager.validate()

            # Update best parameters if current model is better
            if f1_result > best_f1_score:
                best_f1_score = f1_result
                best_params = hparams
                best_model = copy.deepcopy(model)
                best_trial = trial_id

            # Ensure all operations are done
            torch.cuda.synchronize()

        # Testing
        model_manager.test(best_model) 
        
        # Saving
        ModelManager.save(best_model, suffix=f'trial_{best_trial}_best')

        # Logging
        logger.info(f'Best F1 Score: {best_f1_score:.4f}')
        logger.info('Best Hyper-parameters:')
        for param_name, param_value in best_params.items():
            logger.info(f'\t{param_name}: {param_value}')
        return
