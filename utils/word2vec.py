# https://github.com/OlgaChernytska/word2vec-pytorch

from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import WikiText2, WikiText103


def get_english_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer


def get_data_iterator(ds_name, ds_type, root_dir: str = '.data'):
    """
    Get a data iterator for a dataset.
    Args:
        ds_name: Name of the dataset.
        ds_type: Type of the dataset (train, val).
        root_dir: Directory where the dataset is stored.
    """
    if ds_name == "WikiText2":
        data_iter = WikiText2(root = root_dir, split=(ds_type))
    elif ds_name == "WikiText103":
        data_iter = WikiText103(root = root_dir, split=(ds_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    # We use a map-style dataset since the number of obs is small and known
    # The torchtext is by default a iterable-style dataset
    data_iter = to_map_style_dataset(data_iter)
    return data_iter


def build_vocab(data_iter, tokenizer, min_freq = 50):
    """
    Builds vocabulary from iterator
    """
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq=min_freq,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def collate_cbow(batch, text_pipeline, n_window, max_seq_len = 256):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=n_words past words and N=n_window future words.
    
    Long paragraphs will be truncated to contain no more that max_seq_len tokens.
    
    Each element in `batch_input` is N=n_window*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        
        # Apply pre-processing (mainy tokenization)
        text_tokens_ids = text_pipeline(text)

        # Check if text is long enough
        if len(text_tokens_ids) < n_window * 2 + 1:
            continue

        # Truncate long paragraphs
        if max_seq_len:
            text_tokens_ids = text_tokens_ids[:max_seq_len]

        # Move through the batch with a sliding window
        for idx in range(len(text_tokens_ids) - n_window * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + n_window * 2 + 1)]
            # Target word is in the middle of the 2 * window
            output = token_id_sequence.pop(n_window)
            # Context is the rest of the 2 * window
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    # Batch together all the inputs and outputs
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader_and_vocab(
        ds_name,
        ds_type,
        batch_size,
        n_window,
        shuffle=True,
        vocab=None):
    """
    Build a dataloader and vocabulary from a dataset.
    Args:
        ds_name: Name of the dataset.
        ds_type: Type of the dataset (train, val).
        batch_size: Batch size.
        n_window: Number of tokens on each side of the context
        shuffle: Whether to shuffle the dataset.
        vocab: Vocabulary to use.
    """
    data_iter = get_data_iterator(ds_name, ds_type)
    tokenizer = get_english_tokenizer()

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer)
        
    text_pipeline = lambda x: vocab(tokenizer(x))

    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_cbow, 
                           text_pipeline=text_pipeline,
                           n_window=n_window),
    )
    return dataloader, vocab


def get_top_similar(embeddings: np.array,
                    word: str,
                    vocab: Vocab,
                    top_n: int = 10):
    """
    Get top_n similar words to a given word by calculating cosine similarity
    for each word in the vocabulary.
    """
    
    # Create normalized embeddings
    norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
    norms = np.reshape(norms, (len(norms), 1))
    embeddings_norm = embeddings / norms
    
    # Get id of word
    word_id = vocab[word]
    if word_id == 0:
        print("Out of vocabulary word")
        return

    # Similarity
    word_vec = embeddings_norm[word_id]
    word_vec = np.reshape(word_vec, (len(word_vec), 1))
    dists = np.matmul(embeddings_norm, word_vec).flatten()
    top_n_ids = np.argsort(-dists)[1 : top_n + 1]

    top_n_dict = {}
    for sim_word_id in top_n_ids:
        sim_word = vocab.lookup_token(sim_word_id)
        top_n_dict[sim_word] = dists[sim_word_id]
    
    return top_n_dict

