def tokenize_function(examples, tokenizer):
    """
    Function to tokenize text
    """
    return tokenizer(examples['input'], truncation=True)


def chunk_function(examples, block_size):
    """
    Function to chunk text
    """
    # Concatenate all texts via sum of lists
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, we could add padding if the model supported 
    # it instead of this drop, you can customize this part to your needs
    total_length = (total_length // block_size) * block_size
    
    # Split by chunks of max_len size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # Later DataCollatorForLanguageModeling will handling the input, therefore 
    # just copy inputs to labels
    result['labels'] = result['input_ids'].copy()
    
    return result