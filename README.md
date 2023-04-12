# DistilBERT-model-answer-questions-from-PDF

## Background
Nowadays, with the help of ChatPDF, we are able to extract information from large PDF documents easily. However, despite its effectiveness, the inner workings of this tool remain a mystery to most. Therefore, the purpose of this project is to shed light on the workings of ChatPDF, specifically in relation to its application in answering questions from airline industry companies' Corporate and Social Responsibility (CSR) reports.

### Problem
Extracting data from PDF documents was a time-consuming and tedious process, often requiring manual entry or the use of specialized software that may not always be accurate or reliable. With this application, users can get accurate answer quickly.

### Project Objective
Build a model pipeline based on [DistilBERT-base-uncased model](https://huggingface.co/distilbert-base-uncased) that can extract information from PDF documents and answer related questions accurately.

## Data
The data we used is the airline industry companies' Corporate and Social Responsibility (CSR) reports, including Delta, Jetblue, American Airlines, etc.

## Model
The model we used is [DistilBERT-base-uncased model](https://huggingface.co/distilbert-base-uncased).

### Architecture





The reason we used this model are:

1.





## Pipeline

## Critical analysis
- This project revels part of the process of how ChatPDF works. The difference here is that ChatPDF has been trained with a lot of different types of pdf while this DitilBERT model was only retrained with Corporate and social responsibility (CSR) reports data.
- For the next step, we can increase the diversity of the training data so that the model will be able to process different types of pdf and answer all kinds of questions.

## Code demonstration

## References

### Data

### Hugginface model

## Viedo link
