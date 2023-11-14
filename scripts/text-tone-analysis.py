#!/usr/bin/env python3

# Title: text_tone_analysis.py 
# Abstract:
# Author: Will Robbins
# Email: wrobbins@nd.edu 
# Date: 10/23/23 

import os
import sys
import tqdm
import pandas as pd
from scipy.special import softmax  # Added import

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def polarity_scores(input, model, tokenizer):
    # Tokenize user inputted text
    encoded_text = tokenizer(input, return_tensors='pt')

    # Pass the tokenized input through the model to get scores
    output = model(**encoded_text)
    scores = output.logits[0].detach().numpy()
    
    # Apply softmax to convert scores to probabilities
    scores = softmax(scores)
    
    # Create a dictionary to store sentiment scores
    scores_dict = {
        'neg' : scores[0],
        'neu' : scores[1],
        'pos' : scores[2]
    }
    
    return scores_dict

def main():
    # initialize model and tokenizer, here we are using a pre trained model trained on tweets
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    while True:
        user_input = input("Enter a text to classify (or 'exit' to quit): ")
        
        # Check if the user wants to exit the program
        if user_input.lower() == 'exit':
            break

        # Call the polarity_scores function to analyze sentiment
        roberta_result = polarity_scores(user_input, model, tokenizer)

        # Display the sentiment analysis results
        print("Sentiment Analysis Results:")
        print(f"Negative: {roberta_result['neg']:.2f}")
        print(f"Neutral: {roberta_result['neu']:.2f}")
        print(f"Positive: {roberta_result['pos']:.2f}")

if __name__ == '__main__':
    main()
