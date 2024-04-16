import sys
import json
import os
import pandas as pd

from datasets import load_dataset
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

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-df_path", type=str, help="path to the df contain both orginal and LLM data", default="data/data_df.json")
    parser.add_argument("-test_size", type=float, help="proportion of the ORGINAL data to use as test data", default=0.2)
    parser.add_argument("-random_state", type=int, help="random state for train test split", default=42)
    parser.add_argument('-use_LLM_data_train', action='store_true', default=False,help='add this flag to use LLM data in test data, otherwise only original data is used in training')
    #parser.add_argument('-use_LLM_data_test', action='store_true', default=False,help='add this flag to include LLM data in test data, otherwise only original data is used in test data') 
    parser.add_argument('-use_cased_model', action='store_true', default=False,help='add this flag to use cased model, otherwise uncased model is used')
    #parser.add_argument('-subsset', type=int, help='number of samples to use for training', default=None)

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
        dataset_dict['train'] = dataset_dict['train'].concatenate(dataset_dict['LLM_data'])    
    # remove unnecessary keys
    dataset_dict.pop("orginal_data")
    dataset_dict.pop("LLM_data")
    return dataset_dict

def create_debug_data(df_path):
    df = pd.read_json(df_path)
    true_rows = df[df['llm_generated'] == False].head(20)
    false_rows = df[df['llm_generated'] == True].head(20)
    # Step 3: Concatenate the two filtered DataFrames
    result = pd.concat([true_rows, false_rows])
    # save to a new json file
    result.to_json('data/debug_data.json', orient='records', lines=True)      

#make if main statement
if __name__ == '__main__':
    args = parse_command_line_arguments()
    dataset_dict = load_dataset('json', data_files={'all_data': args.df_path})
    splitData(dataset_dict, **vars(args))







