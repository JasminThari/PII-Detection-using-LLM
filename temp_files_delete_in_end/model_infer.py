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


#cuda test
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')


model_folder = "best_model"

text = "Design Thinking for innovation reflexion-Avril 2021-Nathalie Sylla\n\nChallenge & selection\n\nThe tool I use to help all stakeholders finding their way through the complexity of a project is the  mind map.\n\nWhat exactly is a mind map? According to the definition of Buzan T. and Buzan B. (1999, Dessine-moi  l'intelligence. Paris: Les \u00c9ditions d'Organisation.), the mind map (or heuristic diagram) is a graphic  representation technique that follows the natural functioning of the mind and allows the brain's  potential to be released."
text_2 = "Gandhi Institute of Technology and Management   Higher School of Economics\n\nEssay \u21161\n\non Economics course\n\nTopic 7\n\n\u201cWhy are people sometimes altruistic?\u201d\n\nStefano Lovato\n\nMDI-191 student\n\nSathyabama\n\nApril, 2020\n\nIn many classical economic theories, it is common to determine a human, as a very egoistic  creature. Adam Smith\u2019s books (but not all of them)"
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

classifier = pipeline("ner", model=model_folder, tokenizer=tokenizer)
#classifier(text)



# different way to get the predictions
inputs = tokenizer(text, return_tensors="pt")
# inspect the tokens
tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
model = AutoModelForTokenClassification.from_pretrained(model_folder)
with torch.no_grad():
    logits = model(**inputs).logits

predictions = torch.argmax(logits, dim=2)
predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
#predicted_token_class

