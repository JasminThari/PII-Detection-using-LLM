import pandas as pd
import re
import spacy
from spacy.lang.en import English
import string

# Load the English tokenizer from Spacy
spacy_nlp = spacy.load('en_core_web_sm')
eng_tokenizer = English().tokenizer

def tokenize_with_spacy(text, tokenizer=eng_tokenizer):
    """
    Tokenizes a given text using a specified Spacy tokenizer.

    Parameters:
    - text (str): The text to be tokenized.
    - tokenizer: The Spacy tokenizer to use for tokenization. Default is the English tokenizer.

    Returns:
    - dict: A dictionary containing the tokens and a flag indicating trailing whitespace for each token.
    """
    tokenized_text = tokenizer(text)
    tokens = [token.text for token in tokenized_text]
    trailing_whitespace = [bool(token.whitespace_) for token in tokenized_text]
    return {'tokens': tokens, 'trailing_whitespace': trailing_whitespace}

def adjust_token_labels(row):
    """
    Processes a row from a DataFrame to adjust tokens and labels based on specific criteria.
    
    Parameters:
    - row (pd.Series): A row from a DataFrame containing the columns 'tokens', 'labels', and 'trailing_whitespace'.
    
    Returns:
    - pd.Series: A Series with updated document, tokens, trailing_whitespace, and labels.
    """
    tokens, labels, trailing_whitespace = row.tokens, row.labels, row.trailing_whitespace
    new_tokens, new_labels, new_trailing_whitespaces = [], [], []
    # Adjust labels to remove prefix
    labels = [l.split("-")[1] if l != "O" else l for l in labels]
    
    for i in range(len(tokens)):
        t = tokens[i]
        l = labels[i]
        ws = trailing_whitespace[i]
        
        prev_l = labels[i - 1] if i > 0 else "O"
        next_l = labels[i + 1] if (i + 1) < len(labels) else "O"
        
        # Correct mislabeled tokens based on patterns
        if l != "O" and re.search(r'\+\d+', t):
            l = "PHONE_NUM"
        if l == "STREET_ADDRESS" and prev_l == "PHONE_NUM" and prev_l == next_l:
            l = "PHONE_NUM"
        elif l == "STREET_ADDRESS" and l != next_l and l != prev_l:
            l = "O"
        
        # Retokenize and adjust labels for punctuation
        tok_ = tokenize_with_spacy(t)
        spacy_tokens = tok_["tokens"]
        new_tokens.extend(spacy_tokens)
        
        new_labels.extend([l if st not in string.punctuation else "O" for st in spacy_tokens])
        
        new_trailing_whitespaces.extend(tok_["trailing_whitespace"])
        new_trailing_whitespaces[-1] = ws
        
    return pd.Series({"document": row.document, "tokens": new_tokens, "trailing_whitespace": new_trailing_whitespaces, "labels": new_labels})

def refine_punctuation_labels(row):    
    """
    Updates labels for tokens based on specific rules related to punctuation and sequence.

    Parameters:
    - row (pd.Series): A series containing 'tokens' and 'labels'.

    Returns:
    - list: A list of updated labels.
    """
    tokens = row.tokens
    labels = row.labels
    new_labels = ["O"] * len(labels)
    
    for i, (t, l) in enumerate(zip(tokens, labels)):
        prev_l = new_labels[i - 1] if i > 0 else "O"
        next_l = labels[i + 1] if i + 1 < len(labels) else "O"
        
        # Rules for updating labels based on context and punctuation
        if (prev_l == "NAME_STUDENT" or prev_l == "O") and t == "'s":
            new_labels[i] = "O"
        elif t == "(" and next_l == "PHONE_NUM":
            new_labels[i] = next_l
        elif t == ")" and prev_l == "PHONE_NUM":
            new_labels[i] = prev_l
        elif (t in string.punctuation) and (prev_l == next_l) and (prev_l != "O"):
            if t == "," and prev_l != "STREET_ADDRESS":
                new_labels[i] = "O"
            elif t == "." and prev_l == "NAME_STUDENT":
                new_labels[i] = "O"
            else:
                new_labels[i] = prev_l
        else:
            new_labels[i] = l
            
    return new_labels

def create_bio_labels(labels):
    """
    Converts a list of labels into BIO (Beginning, Inside, Outside) format.

    Parameters:
    - labels (list): A list of labels to be converted.

    Returns:
    - list: The list of labels in BIO format.
    """
    new_labels = ["O"] * len(labels)
    prev_l = "O"
    for i, l in enumerate(labels):
        if l != "O":
            if l != prev_l:
                new_labels[i] = "B-" + l
            elif l == prev_l:
                new_labels[i] = "I-" + l
        prev_l = l
    return new_labels