import torch
import argparse
import os
import sys
# import timm
from datasets import load_from_disk

# load tqdm
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer)

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_folder", type=str, help="This folder should contain a subfolder named best_model and tokenized_data. It should either be called 'cased' or 'uncased' depending on the model used", default="trained_models/uncased")
    parser.add_argument("-batch_size", type=int, help="batch size for prediction", default=124)
    args = parser.parse_args()
    return args

def make_predictions(model, tokenizer, test_data, batch_size, device):
        number_of_examples = len(test_data["tokens"])
        predictions = []

        # Initialize tqdm to track progress
        with tqdm(total=number_of_examples) as pbar:
            # Iterate through the data in batches
            for i in range(0, number_of_examples, batch_size):
                # Get the current batch of inputs
                batch_input = tokenizer(
                    test_data["tokens"][i : i + batch_size],
                    padding=True,
                    truncation=True,
                    is_split_into_words=True,
                    max_length=model.config.max_position_embeddings,
                    return_tensors="pt",
                )
                batch_input.to(device)
                # Perform operations within no_grad() context
                with torch.no_grad():
                    # Perform predictions
                    batch_logits = model(**batch_input).logits
                    # Get predictions and convert them to token classes
                    batch_predictions = torch.argmax(batch_logits, dim=2)
                    batch_predicted_token_class = [
                        [model.config.id2label[t.item()] for t in predictions]
                        for predictions in batch_predictions
                    ]

                predictions.extend(batch_predicted_token_class)
                # Update tqdm progress bar
                pbar.update(len(batch_input["input_ids"]))
        return predictions

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')

    args = parse_command_line()
    model_folder = args.model_folder
    #read whether to use cased or uncased model
    if os.path.basename(model_folder) == "uncased":
        cased = False
    elif os.path.basename(model_folder) == "cased":
        cased = True
    else:
        raise ValueError("This folder should contain a subfolder named best_model, and tokenized_data. It should either be called 'cased' or 'uncased' depending on the model used")
    
    batch_size = args.batch_size
 
    # load the data from the disk
    data_folder = os.path.join(model_folder, "tokenized_data")
    data_dict = load_from_disk(data_folder)
    
    # load the model
    model = AutoModelForTokenClassification.from_pretrained(os.path.join(model_folder, "best_model"))
    # mode model to device
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # choose correct tokenizer
    tokenizer_name = "distilbert-base-uncased" if not cased else "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        padding=True,
        truncation=True,
        is_split_into_words=True,
        max_length=512,
        return_tensors="pt",
    )
    # make predictions in batches
    test_data = data_dict["test"]
    # make predictions in batches           
    
    predictions_test = make_predictions(model, tokenizer, test_data, batch_size, device)
    test_data = test_data.add_column("predicted_labels", predictions_test)
    # save the updated data_dict to the disk
    test_data.save_to_disk(os.path.join(model_folder, "predictions_test"))
    # also make predictions on the validation data
    val_data = data_dict["validation"]
    predictions_val = make_predictions(model, tokenizer, val_data, batch_size, device)
    val_data = val_data.add_column("predicted_labels", predictions_val)
    # save the updated data_dict to the disk
    val_data.save_to_disk(os.path.join(model_folder, "predictions_validation"))
