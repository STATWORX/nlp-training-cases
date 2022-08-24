"""
Helper functions
"""

import spacy
import numpy as np
from transformers import AutoTokenizer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def flatten(col):
    """
    Flatten a list of lists using a recursion.
    """
    rt = []
    for i in col:
        if isinstance(i,list): 
            rt.extend(flatten(i))
        else: 
            rt.append(i)
    return rt


def tokenize_and_chunk(
    texts, tokenizer, context_length: int = 128, add_eos: bool = False):
    """
    Tokenize and chunk the texts. 
    Alternative implementation, operating only on the input_ids as suggested by:
    https://www.youtube.com/watch?v=8PmhEIXhBvI

    Args:
        texts: A list of texts.
        tokenizer: A (huggingface) tokenizer
        context_length: The length of the context.
        add_eos: Whether to add the EOS token to the end of each text.
    """
    all_input_ids = []
    
    # Tokenize all pieces of the text and save the input_ids
    seq_of_input_ids = tokenizer(texts['text'])['input_ids']
    
    # We will loop over the long sequence of input_ids
    for input_ids in seq_of_input_ids:
        all_input_ids.extend(input_ids)
        
        # Add the EOS token to the end, if not already present
        if add_eos:
            all_input_ids.append(tokenizer.eos_token_id)

    # Create the chunks with the context_length
    chuncks = []
    for idx in range(0, len(all_input_ids), context_length):
        chuncks.append(all_input_ids[idx: idx + context_length])

    return {'input_ids': chuncks}



class HelperModel:

    def __init__(self, word2vec = "en_core_web_md", bert = "bert-base-cased"):

        self.nlp = spacy.load(word2vec)

        self.tokenizer = AutoTokenizer.from_pretrained(bert)
        self.bert = pipeline('feature-extraction',model=bert, tokenizer=self.tokenizer, device=-1)

        self.vectors = []
        self.words = []
        for key, vector in self.nlp.vocab.vectors.items():

            word_string = self.nlp.vocab.strings[key]
            self.vectors.append(vector)
            self.words.append(word_string)

        self.vectors = np.array(self.vectors)


    def word2vecEmbeddings(self, text):
        tokens = self.nlp(text)
        out_words, out_vectors = [], []
        invalid = []
        for token in tokens:
            if token.is_oov:
                invalid.append(token)
                continue
            out_words.append(token.text)
            out_vectors.append(token.vector)

        if len(invalid) > 0:
            print('No vectors exist for the following words, as they were not in the training vocabulary:')
            print(invalid)
            print('result includes all available vectors')

        return out_words, np.array(out_vectors)

    def getAllWord2Vec(self):
        return self.words, self.vectors

    def bertEmbeddings(self, text):
        # select first list/batch element (equals the input sentence)
        # exlcude the CLS and SEP tokens
        return np.array(self.bert(text)[0][0:-1])

    def cosineSimilarity(self, word1, word2):
        return cosine_similarity(word1[np.newaxis,] , word2[np.newaxis,])

 

    def train_classifier( self, df_train, df_test):
        trainset = TensorDataset(torch.tensor(df_train.embeddings.values.tolist()), torch.tensor(df_train.label.values.tolist())) 
        # train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=False) 

        testset = TensorDataset(torch.tensor(df_test.embeddings.values.tolist()), torch.tensor(df_test.label.values.tolist())) 
        # test_loader = DataLoader(testset, batch_size=128, shuffle=False, drop_last=False) 

        train_indices = list(range(len(trainset)))
        test_indices = list(range(len(testset)))


        layers = [nn.Linear(768, 200), nn.Linear(200, 50), nn.Linear(50, 1),nn.Sigmoid()]
        model = nn.Sequential(*layers).cpu()
        bceloss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_losses = []
        test_losses = []


        def train(model, trainset, optimizer, bceloss):
            model.train()
            train_loss = 0
            shuffle(train_indices)
            for idx in range(0,len(train_indices),128):

                (input, target) = trainset[idx:idx+128]

                optimizer.zero_grad()
                fwd = model(input.cpu().float())
                loss = bceloss(fwd,target.cpu().float().unsqueeze(1))

                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            train_loss /= int(len(train_indices)/128)
            return train_loss

        def test(model, testset,bceloss):
            model.eval()
            test_loss = 0

            with torch.no_grad():
                for idx in range(0,len(test_indices),128):
                    (input, target) = testset[idx:idx+128]
                    fwd = model(input.cpu().float())
                    loss = bceloss(fwd,target.cpu().float().unsqueeze(1))
                    test_loss += loss.item()


            test_loss /= int(len(testset))
            return test_loss


        for epoch in range(8):
            train_loss = train(model, trainset,optimizer, bceloss)
            test_loss = test(model, testset, bceloss)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)

        self.smsmodel = model
        
        return model,train_losses, test_losses

    def predict_sms(self, test_df):
        with torch.no_grad():
            pred = self.smsmodel(torch.tensor(test_df.embeddings.values.tolist()).cpu().float())
        return (pred.numpy() > 0.5).astype(np.int8)
