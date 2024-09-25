"""Data Manager"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import pandas as pd

from loguru import logger
from typing import Any, Tuple
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class DataManager(Dataset):
    """Data Manager"""

    class Config:
        """Data configuration"""
        STATE_SEED = 42
        BATCH_SIZE = 1024

        TRAIN_PORTION = 0.6
        TEST_VALID_SPLIT = 0.5

        FEATURE_TEXT = 'review'
        FEATURE_LABEL = 'sentiment'
        DATA_PATH = "data/processed/IMDB-Dataset-processed.csv"

    def __init__(self, processor: Any = None) -> None:
        """Initialize data manager

        Args:
            processor: Processor object for transforming reviews. Defaults to None.

        Returns:
            None
        """
        self.processor = processor

        self.df = None
        self.input_data = None
        self.target_data = None

        self.X_train = None
        self.y_train = None
        
        self.X_valid = None
        self.y_valid = None

        self.X_test = None
        self.y_test = None

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        return
    
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.input_data)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[str, int]: A tuple containing the review text and its corresponding label.
        """
        text = self.input_data[idx]
        label = self.target_data[idx]
        
        return text, label

    def __load(self) -> None:
        """Load dataset"""
        self.df = pd.read_csv(DataManager.Config.DATA_PATH, header=0)
        
        self.input_data = self.df[DataManager.Config.FEATURE_TEXT]
        self.target_data = self.df[DataManager.Config.FEATURE_LABEL]

        logger.info(f'Data has been loaded from: {DataManager.Config.DATA_PATH}')
        return

    def __preprocess(self) -> None:
        """Preprocess data"""
        if self.processor is not None:
            self.input_data = self.processor.execute(corpus=self.input_data)
        return

    def __split(self) -> None:
        """Split dataset"""
        # Split
        self.X_train, X_valid_test, self.y_train, y_valid_test = train_test_split(
            self.input_data,
            self.target_data,
            train_size=DataManager.Config.TRAIN_PORTION,
            random_state=DataManager.Config.STATE_SEED,
            shuffle=True
        )

        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            X_valid_test,
            y_valid_test,
            test_size=DataManager.Config.TEST_VALID_SPLIT,
            random_state=DataManager.Config.STATE_SEED,
            shuffle=True
        )
            
        # Combine
        self.train_data = pd.DataFrame({DataManager.Config.FEATURE_TEXT: self.X_train, DataManager.Config.FEATURE_LABEL: self.y_train})
        self.valid_data = pd.DataFrame({DataManager.Config.FEATURE_TEXT: self.X_valid, DataManager.Config.FEATURE_LABEL: self.y_valid})
        self.test_data = pd.DataFrame({DataManager.Config.FEATURE_TEXT: self.X_test, DataManager.Config.FEATURE_LABEL: self.y_test})
        return
    
    def __init_loaders(self) -> None:
        """Initialize loaders"""
        self.dataloader_train = DataLoader(self.train_data, batch_size=DataManager.Config.BATCH_SIZE, shuffle=True)
        self.dataloader_valid = DataLoader(self.valid_data, batch_size=DataManager.Config.BATCH_SIZE, shuffle=False)
        self.dataloader_test = DataLoader(self.test_data, batch_size=DataManager.Config.BATCH_SIZE, shuffle=False)
        return

    def prepare(self) -> None:
        """Prepare data"""
        self.__load()
        self.__preprocess()
        self.__split()
        self.__init_loaders()
        return
