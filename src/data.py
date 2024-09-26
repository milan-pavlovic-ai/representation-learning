"""Data Manager"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import random
import spacy
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger
from typing import Any, Tuple
from spacy.cli import download
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_punctuation


class PandasDataset(Dataset):
    """Pandas dataset for Pytorch"""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initailize pandas dataset
        
        Args:
            df (pd.DataFrame): Dataset.
        
        Returns:
            None
        """
        self.df = df
        self.input_data = self.df[DataManager.Config.FEATURE_TEXT]
        self.target_data = self.df[DataManager.Config.FEATURE_LABEL]
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
        if idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range")
        
        text = self.input_data.iloc[idx]
        label = self.target_data.iloc[idx]
        
        return text, label


class DataManager:
    """Data Manager"""

    class Config:
        """Data configuration"""
        STATE_SEED = 21
        BATCH_SIZE = 128

        TRAIN_SAMPLES = 20_000
        VALID_SAMPLES = 5_000
        TEST_SAMPLES = 25_000

        FEATURE_TEXT = 'review'
        FEATURE_LABEL = 'sentiment'
        DATA_PATH = "data/processed/IMDB-Dataset-processed.csv"

    def __init__(self, sample_size: float = None, processor: Any = None) -> None:
        """Initialize data manager

        Args:
            sample_size (float): Sample size of the dataset.
            processor: Processor object for transforming reviews. Defaults to None.

        Returns:
            None
        """
        self.sample_size = sample_size
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

        # Set reproducibilty
        torch.manual_seed(DataManager.Config.STATE_SEED)
        np.random.seed(DataManager.Config.STATE_SEED)
        random.seed(DataManager.Config.STATE_SEED)
        return

    def __load(self) -> None:
        """Load dataset"""
        self.df = pd.read_csv(DataManager.Config.DATA_PATH, header=0)
        
        if self.sample_size is not None:
            self.df = self.df.sample(frac=self.sample_size, random_state=DataManager.Config.STATE_SEED).reset_index(drop=True)
        
        self.input_data = self.df[DataManager.Config.FEATURE_TEXT]
        self.target_data = self.df[DataManager.Config.FEATURE_LABEL]

        logger.warning(f'Data has been loaded from: {DataManager.Config.DATA_PATH}')
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
            train_size=DataManager.Config.TRAIN_SAMPLES / len(self.input_data),
            random_state=DataManager.Config.STATE_SEED,
            stratify=self.target_data,
            shuffle=True
        )

        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            X_valid_test,
            y_valid_test,
            test_size=DataManager.Config.TEST_SAMPLES / len(X_valid_test),
            random_state=DataManager.Config.STATE_SEED,
            stratify=y_valid_test,
            shuffle=True
        )

        # Combine
        self.train_data = pd.DataFrame({DataManager.Config.FEATURE_TEXT: self.X_train, DataManager.Config.FEATURE_LABEL: self.y_train})
        logger.info(f'Training set: {len(self.train_data)} or {len(self.train_data) / len(self.input_data) * 100:.2f}%')
        
        self.valid_data = pd.DataFrame({DataManager.Config.FEATURE_TEXT: self.X_valid, DataManager.Config.FEATURE_LABEL: self.y_valid})
        logger.info(f'Validation set: {len(self.valid_data)} or {len(self.valid_data) / len(self.input_data) * 100:.2f}%')

        self.test_data = pd.DataFrame({DataManager.Config.FEATURE_TEXT: self.X_test, DataManager.Config.FEATURE_LABEL: self.y_test})
        logger.info(f'Testing set: {len(self.test_data)} or {len(self.test_data) / len(self.input_data) * 100:.2f}%')
        return
    
    def __init_loaders(self) -> None:
        """Initialize loaders"""
        self.dataloader_train = DataLoader(PandasDataset(self.train_data), batch_size=DataManager.Config.BATCH_SIZE, shuffle=True, pin_memory=True)
        self.dataloader_valid = DataLoader(PandasDataset(self.valid_data), batch_size=DataManager.Config.BATCH_SIZE, shuffle=False, pin_memory=True)
        self.dataloader_test = DataLoader(PandasDataset(self.test_data), batch_size=DataManager.Config.BATCH_SIZE, shuffle=False, pin_memory=True)
        return

    def prepare(self) -> None:
        """Prepare data"""
        self.__load()
        self.__preprocess()
        self.__split()
        self.__init_loaders()
        return

    def into_tokens_lists(self) -> None:
        """Transform everything into tokens list"""
        self.X_train = self.X_train.apply(lambda x: x.split())
        self.X_valid = self.X_valid.apply(lambda x: x.split())
        self.X_test = self.X_test.apply(lambda x: x.split())

        self.train_data[DataManager.Config.FEATURE_TEXT] = self.train_data[DataManager.Config.FEATURE_TEXT].apply(lambda x: x.split())
        self.valid_data[DataManager.Config.FEATURE_TEXT] = self.valid_data[DataManager.Config.FEATURE_TEXT].apply(lambda x: x.split())
        self.test_data[DataManager.Config.FEATURE_TEXT] = self.test_data[DataManager.Config.FEATURE_TEXT].apply(lambda x: x.split())

        # Re-initialized data
        self.__init_loaders()
        return

class TextPreprocessor:
    """Text preprocessor"""

    class Config:
        """Preprocessor Configuration"""
        ENGLISH_CORPUS = 'en_core_web_sm'
        CACHED_PROCESSED_DATA_PATH = 'data/processed/IMDB-Dataset-processed-in-details.csv'
        MIN_SIZE_TOKEN = 2

    def __init__(self) -> None:
        """Initialize the text preprocessor"""
        try:
            self.nlp = spacy.load(TextPreprocessor.Config.ENGLISH_CORPUS)
        except OSError:
            logger.info(f"Downloading {TextPreprocessor.Config.ENGLISH_CORPUS} model...")
            download(TextPreprocessor.Config.ENGLISH_CORPUS)
            self.nlp = spacy.load(TextPreprocessor.Config.ENGLISH_CORPUS)

        self.default_filters = [
            lambda doc: doc.lower(),    # Lowercase
            strip_punctuation,          # Remove puncts
            remove_stopwords,           # Remove stop words
            lambda doc: ' '.join([
                token for token in doc.split() 
                if (len(token) >= TextPreprocessor.Config.MIN_SIZE_TOKEN or token.isdigit())  # Keep tokens with len >= 2 or digits
            ]),
            lambda doc: ' '.join([token.lemma_ for token in self.nlp(doc)]),    # Lemmatization
            # stem_text                   # Stemming
        ]
        return
    
    def execute(self, corpus: pd.Series) -> pd.Series:
        """Preprocess corpus

        Args:
            corpus (pd.Series): List of documents.

        Returns:
            pd.Series: Preprocessed List of documents.
        """
        if os.path.exists(TextPreprocessor.Config.CACHED_PROCESSED_DATA_PATH):
            processed_corpus = pd.read_csv(TextPreprocessor.Config.CACHED_PROCESSED_DATA_PATH, header=0)
            processed_corpus = pd.Series(processed_corpus.iloc[:, 0]) 
            logger.warning(f'Load pre-processed corpus from {TextPreprocessor.Config.CACHED_PROCESSED_DATA_PATH}')
            return processed_corpus
        
        processed_docs = []
        for doc in tqdm(corpus, desc='Preprocessing data'):
            tokens = preprocess_string(doc, filters=self.default_filters)
            processed_docs.append(' '.join(tokens))

        processed_corpus = pd.Series(processed_docs)
        processed_corpus.to_csv(TextPreprocessor.Config.CACHED_PROCESSED_DATA_PATH, index=False)
        logger.info(f'Corpus has been pre-processed and saved at {TextPreprocessor.Config.CACHED_PROCESSED_DATA_PATH}')
        
        return processed_corpus
