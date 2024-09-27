# Representation Learning

## Overview
**Representation learning** is a subset of machine learning that focuses on learning how to represent input data. The goal is to transform raw data into a format that makes it easier for machine learning algorithms to make predictions.

In NLP, **textual representations** can capture syntax, semantics, and context, which are all essential for tasks like sentiment analysis, machine translation, text summarization, etc.

## Types
Representation learning can be categorized based on:
- **Learning Type**: Supervised, Unsupervised, Semi-Supervised, and Self-Supervised learning.
- **Data Format**: Tabular, Graph, Text, Audio, Image, Video, etc. 

This repository focuses on **text** data.

## Text Representation
This repository implements various methods for learning text representations, each suited for different tasks and goals in NLP. Below is a summary of the text representation methods implemented:

### 1. **TF-IDF** (Term Frequency-Inverse Document Frequency)
TF-IDF converts text into vectors by measuring the importance of words in a document relative to the entire corpus, enhancing document classification and information retrieval.

**Citation**: 
Salton, G., & Buckley, C. (1988). [Link](https://doi.org/10.1016/0306-4573(88)90021-0)

### 2. **Word2Vec** (Word to Vector)
Word2Vec generates dense vector embeddings by predicting surrounding words based on context, capturing semantic relationships in text.

**Citation**: 
Mikolov, T. et al. (2013). [Link](https://arxiv.org/abs/1301.3781)

### 3. **LSTM** (Long Short-Term Memory)
LSTMs are a type of RNN designed to handle sequential data, making them effective for tasks like sentiment analysis by maintaining long-term dependencies.

**Citation**: 
Hochreiter, S., & Schmidhuber, J. (1997). [Link](https://doi.org/10.1162/neco.1997.9.8.1735)

### 4. **BERT** (Bidirectional Encoder Representations from Transformers)
BERT reads text bidirectionally, capturing context from both sides, and is well-suited for various NLP tasks through fine-tuning.

**Citation**: 
Devlin, J. et al. (2019). [Link](https://arxiv.org/abs/1810.04805)

### 5. **GPT** (Generative Pretrained Transformer)
GPT is trained to generate text by predicting one token at a time, excelling in generative tasks like text completion and summarization.

**Citation**: 
Radford, A. et al. (2019). [Link](https://cdn.openai.com/transcripts/gpt-2.pdf)

### 6. **BART** (Bidirectional and Auto-Regressive Transformers)
BART combines a bidirectional encoder with an autoregressive decoder, making it effective for both understanding and generating text.

**Citation**: 
Lewis, M. et al. (2020). [Link](https://arxiv.org/abs/1910.13461)

### 7. **T5** (Text-to-Text Transfer Transformer)
T5 treats every NLP problem as a text-to-text task, enabling versatile applications across various NLP challenges.

**Citation**: 
Raffel, C. et al. (2020). [Link](https://arxiv.org/abs/1910.10683)


## Project Structure

### Data Module (data.py)

The data module handles:
- Loading and splitting the dataset
- Preparing the dataset for batch processing
- Preprocessing the text (removing stop words, lemmatization, etc)

### Model Module (model.py)

The model module is responsible for:
- Managing models (training, evaluating, saving, loading, etc)
- Model selection and algorithm optimization
- Definition of the `TextRepresentation` abstract class, which all text representation methods implements
- Implementing a linear classifier used across all appraoches
- Definition of the `TextClassifier`, which combines text representation and linear classifier for sentiment classification

### Representation methods

In the `represents` folder, there are implementations for various text representation methods:
- **TF-IDF**, **Word2Vec**, **LSTM**, **BERT**, **GPT**, **BART**, and **T5**

These modules demonstrate different approaches to text representation, showcasing their effectiveness in sentiment classification task. There is also an option for state-of-the-art methods to use only pretrained representations or to fine-tune them.

### Dataset

The raw dataset can be found in the `raw/processed` directory. The processed dataset can be found in the `data/processed` directory.

The dataset is sourced from [Kaggle: Large Movie Review Dataset (Maas et al., 2011)](https://www.kaggle.com/datasets/hamditarek/imdb-dataset-50k-maas-et-al-2011), focusing on classifying movie reviews as positive or negative.


## Next steps

- **Expansion of Methods**: Additional text representation methods
- **Diverse Problem Types**: Tackle different types of problems
- **Automated Comparison Scripts**: Implementing automated scripts for comparing representation methods would streamline experimentation


## Setup

Ensure you have **Python 3.11** installed. This project is compatible with CUDA, so if you have an NVIDIA GPU, it will be automatically detected for accelerated computation.

### Requirements
- **Git**: Version control system
- **Poetry**: Dependency management tool for Python
- **Python 3.11**: Programming language version

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/milan-pavlovic-ai/representation-learning.git
   ```

2. Install dependecies in the virtual environment

   ```bash
   poetry install
   ```
3. Activate virtual environment

   ```bash
   poetry shell
   ```


## Contribution

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Feel free to open issues for bug reports, feature requests, or discussions.
