import sys
import json
import os
import pandas as pd
from omegaconf import OmegaConf

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
from tqdm import tqdm 

from data_prep_for_model import parse_command_line_arguments, prepData


if __name__ == "__main__":
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
   
    # create folder for the model and prepare data:
    command_line_args = parse_command_line_arguments() # should contain the path to the config file
    config = OmegaConf.load(command_line_args.config_path)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #output_dir = os.path.join("trained_models",config.name,current_time)
    #os.makedirs(output_dir, exist_ok=True)
    # prepdata will save the tokenized data in the correct folder.
    # it will also save a copy of the config file in the same folder
    print("I am in the new script")
    data_dict = prepData(config,model_path=current_time)

    # now we can train the model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name) 

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    seqeval = evaluate.load("seqeval")

    label2id = dict(config.data_prep.label2id)

    id2label = {v: k for k, v in label2id.items()}
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
    #model_name = config.model_name
    model = AutoModelForTokenClassification.from_pretrained(config.model_name, num_labels=len(label2id.keys()), id2label=id2label, label2id=label2id)
    # move the model to the device
    model.to(device)

    # define the training arguments
    output_dir = os.path.join("trained_models",config.name,current_time)
    training_args = TrainingArguments(
        output_dir=output_dir,**config.training_args)

    if config.weighted_loss:
        # find the label distrubution in the training data
        train_labels = torch.tensor(data_dict["train"]["labels"]).to(device)
        label_distribution = torch.bincount(train_labels[train_labels >= 0].flatten(), minlength=len(label_list))
        num_total_labels = label_distribution.sum()
        # create the weight tensor
        weight_tensor = 1 / label_distribution.float() #potential division by zero leaing to inf values
        # replace the infinities with a small number
        weight_tensor[weight_tensor == float("inf")] = 1e-8   
  
        #class to use if the weighted loss function is chosen
        class WeightedLossTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=data_dict["train"],
            eval_dataset=data_dict["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,            
        )

    else:
 
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=data_dict["train"],
            eval_dataset=data_dict["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            
            
        )

    trainer.train()

    # save the model
    model.save_pretrained(os.path.join(output_dir, "best_model"))




