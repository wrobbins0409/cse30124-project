#!/usr/bin/env python3

# Title: sentiment_analysis_train.py 
# Abstract:
# Author: Will Robbins
# Email: wrobbins@nd.edu 
# Estimate:
# Date: 11/15/23 

import os
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path
from datasets import load_dataset, load_metric
from huggingface_hub import login
import numpy as np

# # get token for user to upload
token = input("Enter hugging face token to login: ")

# # logging into hugging face to access datasets and upload model
login(token)

# will be using glue database and training on SST-2 (Stanford Sentiment Treebank) Determine if the sentence has a positive or negative sentiment.

# set task
task = "sst2"

# # set model head for fine tuning, in this case using distilbert-base-uncased due to the fact it is mostly intended for downstream fine tuning which is good for our objective
model_head = "distilbert-base-uncased"

# set batch size depending on gpu
batch_size = 8

# call load_dataset to cache dataset of data for sentiment analysis from glue dataset
dataset = load_dataset("glue", task)

# call load metric to get a metric object that we can use to examine our dataset
metric = load_metric("glue", task)

# show dataset
print(f'\nDataset format: \n\n{dataset}\n')

# load a tokenizer that is based on the model we will be fine tuning, in this case it will be a DistilBertTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_head, use_fast=True)

# test tokenizer
print(f'tokenizer output: {tokenizer.convert_ids_to_tokens(tokenizer.encode(dataset["train"]["sentence"][0]))}')
print(f'Encoded tokens:   {tokenizer.encode(dataset["train"]["sentence"][0])}\n')

# define prepocessor function to apply tokenizer to all lines of each part of the data set, that being the training data, validation data, and testing data
def preprocess_function(data):
    return tokenizer(data["sentence"], truncation=True)

# apply function to dataset with map method of dataset
processed_dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)

# instantiate model from our chosen models head and use 2 labels because we are utilizing sst2 for sentiment analysis
model = AutoModelForSequenceClassification.from_pretrained(model_head, num_labels=2)

model_name = input("Enter name for output directory of model: ")

# # need to create a training arguments object with our specified configuration, using our specified batch size and testing our model for accuracy
args = TrainingArguments(
    model_name,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# we need to create a function that computes the metrics based on the evaluation dataset
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# instantiate our trainer object with our model, training dataset, validation dataset, and our args object
trainer = Trainer(
    model,
    args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=tokenizer,

    # need to include this so that the trainer knows how to compute its metrics
    compute_metrics=compute_metrics
)

# actually train the model
trainer.train()

# evaluate our fine tuned model
trainer.evaluate()

# upload to hugging face
trainer.push_to_hub()
            



