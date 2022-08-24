"""
Test script checking main dependencies
"""

# Test all imports once
import transformers
import tokenizers
import torch
import torchvision
import pandas
import numpy
import huggingface_hub
import bitsandbytes
import datasets
import fastpunct
import editdistance
import nltk
import spacy
import wmd
import dask
import keybert
import phonemizer
import pytorch_lightning
import torchtext
import torchdata
import category_encoders
import s3fs
import sklearn
import matplotlib
import bertviz
import gensim

from transformers import pipeline
from datasets import load_dataset

# Test torch installation
print(f'GPU available: {torch.cuda.is_available()}')

x = torch.normal(0, 1, (4, 4))
y = torch.normal(0, 1, (4, 4))
r = torch.matmul(x, y)

print(f'Shape of r: {r.shape}')

# Test transfromers package
# Pipeline will use distilbert-base-cased-distilled-squad as default
question_answerer = pipeline("question-answering")
result = question_answerer(
    {
        "question": "What is the name of the repository ?",
        "context": "Pipeline has been included in the huggingface/transformers repository",
    }
)

print(f'Test of transformer pipeline: \n {result}')

# Load a slice from IMDB dataset
data = load_dataset('imdb', split='train[:10]')
print(f'Data Shape of IMDB: {data.shape}')

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

print(f'Spacy loaded with language: {nlp.lang}')

print('All tests passed')
