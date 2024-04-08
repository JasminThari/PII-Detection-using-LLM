import re
from collections import Counter
import math
import pandas as pd
import matplotlib.pyplot as plt
# NLP
import spacy
from spacy.lang.en import English
spacy_nlp = spacy.load('en_core_web_sm')
eng_tokenizer = English().tokenizer

# Text Processing and Analysis
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def compute_tf(word_dict, document):
    """
    This function calculates the Term Frequency (TF) for each word in a document based on the word count in the document.

    Args:
    - word_dict: Dictionary containing word counts for each word in the document.
    - document: List of words in the document.

    Returns:
    A dictionary with words as keys and their TF values as values.
    """
    tf_dict = {}
    doc_length = len(document)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(doc_length)
    return tf_dict

def compute_idf(documents):
    """
    This function calculates the Inverse Document Frequency (IDF) for each unique word in a collection of documents.

    Args:
    - documents: List of documents, where each document is represented as a list of words.

    Returns:
    A dictionary with words as keys and their IDF values as values.
    """
    N = len(documents)

    # Create a set of all unique words
    all_words = set()
    for doc in documents:
        all_words = all_words.union(set(doc))

    # Initialize IDF dictionary with zeros for each word
    idf_dict = dict.fromkeys(all_words, 0)

    # Count the number of documents that contain each word
    for word in idf_dict.keys():
        idf_dict[word] = sum(word in doc for doc in documents)

    # Calculate IDF
    for word, val in idf_dict.items():
        idf_dict[word] = math.log(N / float(val))

    return idf_dict

def compute_tfidf(tf_doc, idfs):
    """
    This function calculates the TF-IDF (Term Frequency-Inverse Document Frequency) for each word in a document.

    Args:
    - tf_doc: Dictionary containing the TF values for each word in the document.
    - idfs: Dictionary containing the IDF values for each unique word in the collection of documents.

    Returns:
    A dictionary with words as keys and their TF-IDF values as values.
    """
    tfidf = {}
    for word, val in tf_doc.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def calc_td_idf(all_documents): 
    """
    This function calculates the TF-IDF for each document in a collection of documents.

    Args:
    - all_documents: List of documents, where each document is represented as a list of words.

    Returns:
    A list of dictionaries, where each dictionary represents the TF-IDF values for a document in the collection.
    """
    # tf
    tf_documents = []
    for doc in all_documents:
        word_counts = Counter(doc)
        tf_documents.append(compute_tf(word_counts, doc))
    
    # idf
    idfs = compute_idf([Counter(doc) for doc in all_documents])
    
    # td idf
    tfidf_documents = [compute_tfidf(doc, idfs) for doc in tf_documents]
    return tfidf_documents


def preprocess_texts(data_df):
    """
    This function preprocesses texts in a DataFrame by performing the following steps:
    1. Convert each text to lowercase.
    2. Remove non-alphabetic characters and split the text into words.
    3. Remove stopwords using the English language stopwords set.
    4. Lemmatize the remaining words using WordNet lemmatizer.

    Args:
    - data_df: DataFrame with a 'text' column containing texts to be preprocessed.

    Returns:
    A single string containing the preprocessed and lemmatized words from the input texts.
    """

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    processed_texts = []

    for document in data_df['text']: 
        # Convert to lowercase
        text = document.lower() 
        # Remove non-alphabetic characters and split into words
        words = re.findall(r'\b[a-z]+\b', text)
        # Remove stopwords and lemmatize the words
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        # Join the lemmatized words into a single string and add it to the list
        processed_text = ' '.join(lemmatized_words)
        processed_texts.append(processed_text)

    return processed_texts



def plot_wordcloud(title, tfidf_dict):
    wordcloud = WordCloud(
        collocations=False,
        max_font_size=40,
        random_state=42)

    # Generate the word cloud
    wordcloud.generate_from_frequencies(tfidf_dict)

    # Plot the word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize = 16, fontweight="bold")
    plt.show()
