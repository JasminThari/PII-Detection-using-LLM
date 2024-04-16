import sys
import json
import os
import pandas as pd
from omegaconf import OmegaConf

from datasets import load_dataset,concatenate_datasets
from transformers import (AutoTokenizer, 
                          AutoModelForTokenClassification, 
                          DataCollatorForTokenClassification, 
                          TrainingArguments, 
                          Trainer,
                          
                          pipeline)
import evaluate
import torch
import numpy as np
import datetime
import argparse

# def parse_command_line_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-df_path", type=str, help="path to the df contain both orginal and LLM data", default="data/data_df.json")
#     parser.add_argument("-test_size", type=float, help="proportion of the ORGINAL data to use as test data", default=0.2)
#     parser.add_argument("-random_state", type=int, help="random state for train test split", default=42)
#     parser.add_argument('-use_LLM_data_train', action='store_true', default=False,help='add this flag to use LLM data in test data, otherwise only original data is used in training')
#     #parser.add_argument('-use_LLM_data_test', action='store_true', default=False,help='add this flag to include LLM data in test data, otherwise only original data is used in test data') 
#     parser.add_argument('-use_cased_model', action='store_true', default=False,help='add this flag to use cased model, otherwise uncased model is used')
#     #parser.add_argument('-subsset', type=int, help='number of samples to use for training', default=None)
#     parser.add_argument('-max_length', type=int, help='max length of the tokenized data', default=512)
#     parser.add_argument('-chunk_data', action='store_true', default=False,help='add this flag to chunk the data into smaller pieces')
# 
#     args = parser.parse_args()
#     return args

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, help="path to the config file to use", default=None)
    args = parser.parse_args()
    return args

def splitData(dataset_dict,**kwargs):
    dataset_dict["orginal_data"] = dataset_dict["all_data"].filter(lambda example: example['llm_generated'] == False)
    dataset_dict["LLM_data"] = dataset_dict["all_data"].filter(lambda example: example['llm_generated'] == True)
    # remove the the entry with all data
    dataset_dict.pop("all_data")
    # now create test set and adjust the orginal data to be the training data
    # train and test split based on org data
    train_test = dataset_dict["orginal_data"].train_test_split(test_size=kwargs['test_size'], shuffle=True, seed=kwargs['random_state'])
    # update the dataset_dict with the train and test data
    dataset_dict["train"] = train_test['train']
    dataset_dict["test"] = train_test['test']
    # add LLM data to train data based on the flag
    if kwargs['use_LLM_data_train']:
        dataset_dict['train'] = concatenate_datasets([dataset_dict['train'], dataset_dict['LLM_data']])
        
    # assert check if test and train data are disjoint
    # remove unnecessary keys / data sets       
    dataset_dict.pop("orginal_data")
    dataset_dict.pop("LLM_data")
    dataset_dict = dataset_dict.map(lambda x: {'labels_int': [kwargs["label2id"][i] for i in x['labels']]}) 
    return dataset_dict

def addIntLabels(dataset_dict):
    # dictionary to map the labels to integers
    label2id = {
    "O": 0,
    "B-NAME_STUDENT": 1,
    "I-NAME_STUDENT": 2,
    "B-ID_NUM": 3,
    "I-ID_NUM": 4,
    "B-PHONE_NUM": 5,
    "I-PHONE_NUM": 6,
    "B-EMAIL": 7,
    "I-EMAIL": 8,
    "B-URL_PERSONAL": 9,
    "I-URL_PERSONAL": 10,
    "B-STREET_ADDRESS": 11,
    "I-STREET_ADDRESS": 12,
    "B-USERNAME": 13,
    "I-USERNAME": 14}

    dataset_dict = dataset_dict.map(lambda x: {'labels_int': [label2id[i] for i in x['labels']]})                        
    return dataset_dict

def tokenize_and_align_labels(examples, tokenizer=None, chunk_data=False, max_length=512):
    if tokenizer is None:
        print("tokenizer is None, please provide a tokenizer to tokenize the data")
        return
    
    if chunk_data:
        tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True)
    else: 
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True, max_length=max_length)

    labels = []
    for i, label in enumerate(examples[f"labels_int"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100. # -100 is the default value for ignore_index in CrossEntropyLoss.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def chunk_data(examples,max_length=512):
    """
    This function should on be used if the data has already been tokenized and aligned
    """
    # Currently we just split when the tokenized data is longer than the chunk_size.
    # some smarter splitting might be deployed in the future
    # currently not implemented


def create_debug_data(df_path):
    df = pd.read_json(df_path)
    true_rows = df[df['llm_generated'] == False].head(20)
    false_rows = df[df['llm_generated'] == True].head(20)
    # Step 3: Concatenate the two filtered DataFrames
    result = pd.concat([true_rows, false_rows])
    # save to a new json file
    result.to_json('data/debug_data.json', orient='records', lines=True)


def prepData(config,model_path="non_trained_model"):
    #config = OmegaConf.load(config_path)
    args = config.data_prep    
    dataset_dict = load_dataset('json', data_files={'all_data': args.df_path})
    dataset_dict = splitData(dataset_dict, test_size=args.test_size, random_state=args.random_state, use_LLM_data_train=args.use_LLM_data_train, label2id=args.label2id)
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)      
    # tokenize and align the labels
    dataset_dict = dataset_dict.map(tokenize_and_align_labels,fn_kwargs={'tokenizer':tokenizer, 'chunk_data':args.chunk_data, 'max_length':args.max_length}, batched=True)
    
    # if chunk data is True, chunk the data
    if args.chunk_data:
        raise NotImplementedError("chunking the data is not implemented yet")
    
    # split training into train and validation
    train_val = dataset_dict["train"].train_test_split(test_size=args.val_size, shuffle=True, seed=args.random_state)
    # update the dataset_dict with the train and test data
    dataset_dict["train"] = train_val['train']
    dataset_dict["validation"] = train_val['test']

    # save the tokenized data
    #config_name = os.path.basename(config_path).split('.')[0]
    # make a folder to save the tokenized data
    dir_name = f'trained_models/{config.name}/{model_path}'
    os.makedirs(dir_name, exist_ok=True)
    dataset_dict.save_to_disk(os.path.join(dir_name, 'tokenized_data'))
    # also save the config file
    OmegaConf.save(config, os.path.join(dir_name, 'used_config.yaml')) 
 
# #make if main statement
if __name__ == '__main__':
    #args = parse_command_line_arguments()
    args = parse_command_line_arguments()
    if args.config_path is None:
        print("Please provide a path to the config file")
        sys.exit(1)
    config = OmegaConf.load(args.config_path)
    prepData(config)
    


    
   







