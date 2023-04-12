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
![The-DistilBERT-model-architecture-and-components](https://user-images.githubusercontent.com/89158696/231534143-2dfa1c1e-0cf6-4b57-9253-174fe1ec5151.png)

DistilBERT model is a distilled form of the BERT model. As we can see from the above architecture plot, the distilBERT model adds a knowledge distillation process to reduce the number of transformer layers and the size of each layer. Thus, the DistilBERT model requires less size of datasets and computational resources to train, which is friendly for personal laptops. Although the distilBERT model reduces the size of BERT model by 40%, the distilBERT model still retains 97% of BERT's language understanging abilities and being 60% faster than the BERT model.

As for the DistilBERT-base-uncased model, it's an uncased version of DistilBERT model. In another words, there is no diference between 'english' and 'English' for this model.

And the DistilBERT-base-uncased model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. But for text generation tasks, the GPT model has better performance. Considering that our main task is question answering, we chose the DistilBERT-base-uncased model.



## Pipeline

## Critical analysis
- This project revels part of the process of how ChatPDF works. The difference here is that ChatPDF has been trained with a lot of different types of pdf while this DitilBERT model was only retrained with Corporate and social responsibility (CSR) reports data.
- For the next step, we can increase the diversity of the training data so that the model will be able to process different types of pdf and answer all kinds of questions.

## Code demonstration

## References

### Data

### Hugginface model

## Viedo link
