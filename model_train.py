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

#cuda test
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

#create a folder to save trainied models, if it doesn't exist
os.makedirs("trained_models", exist_ok=True)
# get current time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("trained_models", current_time)
# make a folder with the current time in the trained_models folder
os.makedirs(output_dir, exist_ok=True)
   
# Path to your JSON file
path_to_json = "data/train.json"

# Load the dataset
dataset = load_dataset('json', data_files={'train': path_to_json})

# use the following dictionary to map strings to integers
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
    "I-USERNAME": 14,
}

# reverse the dictionary label2id
id2label = {v: k for k, v in label2id.items()}

# use the dict to map the strings to integers
dataset = dataset.map(lambda x: {'labels_int': [label2id[i] for i in x['labels']]})

# Split the dataset into a training, validation and test dataset
data_dict = dataset['train'].train_test_split(test_size=0.1)
 

#reduce size of both train, validation and test datasets
data_size = None

if data_size is not None:
    data_dict["train"] = data_dict["train"].select(range(data_size))
    #data_dict["validation"] = data_dict["validation"].select(range(data_size))
    data_dict["test"] = data_dict["test"].select(range(data_size))


# find the labels
#label_list = wnut["train"].features["ner_tags"].feature.names

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased") # might be better to used the cased once, since we are doin NER

# function that tokenizes according the chosen model and aligns the labels with the new tokens

def tokenize_and_align_labels(examples):
    #tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True, max_length=512)


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

    #tokenized_inputs["transformed_labels"] = labels
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# we do not need to do this here, is done in the trainer
#tokenized_data_dict = data_dict.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")

label_list = list(label2id.keys())

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=len(label2id.keys()), id2label=id2label, label2id=label2id)
# move the model to the device
model.to(device)

# make a folder to save the model in
#model_name = "ner_pii_model"
#os.makedirs("work3/s204090/", exist_ok=True)

#dir_to_save_model = "work3/s204090"

#output_dir = os.path.join(dir_to_save_model, model_name)

# define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=32, # 32 works on 32gb gpu and so does 64, chech hugginface for model usage
    per_device_eval_batch_size=32,
    num_train_epochs=50,
    #num_train_epochs=25,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2, # limit the total amount of checkpoints, it will delete the older checkpoints
    load_best_model_at_end=True,
    push_to_hub=False
    #save_steps=1000,
                        )

# define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=data_dict["train"].map(tokenize_and_align_labels, batched=True),
    eval_dataset=data_dict["test"].map(tokenize_and_align_labels, batched=True),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# save the model
model.save_pretrained(os.path.join(output_dir, "best_model"))




