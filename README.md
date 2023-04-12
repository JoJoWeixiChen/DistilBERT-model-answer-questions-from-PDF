# DistilBERT-model-answer-questions-from-PDF

## Background
Nowadays, with the help of ChatPDF, we are able to extract information from large PDF documents easily. However, despite its effectiveness, the inner workings of this tool remain a mystery to most. Therefore, the purpose of this project is to shed light on the workings of ChatPDF, specifically in relation to its application in answering questions from airline industry companies' Corporate and Social Responsibility (CSR) reports.

### Problem
Extracting data from PDF documents was a time-consuming and tedious process, especially when the PDF documents are extremlly long. This often requires manual entry or the use of specialized software that may not always be accurate or reliable. However, with the application like ChatPDF and our model, users can get accurate answer from the PDF documents quickly.

### Project Objective
Build a model pipeline based on [DistilBERT-base-uncased model](https://huggingface.co/distilbert-base-uncased) that can understand text-based questions and extract information from PDF documents and answer questions accurately.

## Data
The data we used is the airline industry companies' Corporate and Social Responsibility (CSR) reports, including Delta, Jetblue, American Airlines, etc. 

## Model
The model we used is [DistilBERT-base-uncased model](https://huggingface.co/distilbert-base-uncased).

### Architecture
![The-DistilBERT-model-architecture-and-components](https://user-images.githubusercontent.com/89158696/231534143-2dfa1c1e-0cf6-4b57-9253-174fe1ec5151.png)

DistilBERT model is a distilled form of the BERT model. As we can see from the above architecture plot, the distilBERT model adds a knowledge distillation process to reduce the number of transformer layers and the size of each layer. Thus, the DistilBERT model requires less size of datasets and computational resources to train, which is friendly for personal laptops. Although the distilBERT model reduces the size of BERT model by 40%, the distilBERT model still retains 97% of BERT's language understanging abilities and being 60% faster than the BERT model.

As for the DistilBERT-base-uncased model, it's an uncased version of DistilBERT model. In another words, there is no diference between 'english' and 'English' for this model. And the DistilBERT-base-uncased model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. But for text generation tasks, the GPT model has better performance. Considering that our main task is question answering, we chose the DistilBERT-base-uncased model and fine-tuned it for this project.

For the fine-tuned version of model, once we provided the questions and contexts that contained the answers, it could return the answers in 10 seconds.
<img width="930" alt="Screen Shot 2023-04-12 at 1 32 05 PM" src="https://user-images.githubusercontent.com/89158696/231552239-9d7fb61f-b702-4969-9082-a77a94155311.png">


## Pipeline
The DistilBERT-base-uncased model cannot directly process PDF documents, so we create a pipeline to precess the data and get the answers. The model pipeline including several steps:

### Step 1. Extract data from PDF documents
In this step, we used [Adobe API](https://github.com/adobe/pdfservices-python-sdk-samples/tree/main/src/extractpdf) to extract all the context information and page information from PDF and saved into json files.

<img width="588" alt="Screen Shot 2023-04-12 at 12 52 06 PM" src="https://user-images.githubusercontent.com/89158696/231547606-c774f7de-d104-48fe-a8bd-051afb20bb0a.png">

Then, we transferred the json file into table, and each line saved a piece of context from the PDF document. Also, we truncated the long context information into separate smaller one.

<img width="423" alt="Screen Shot 2023-04-12 at 1 20 00 PM" src="https://user-images.githubusercontent.com/89158696/231566717-3e9724a4-c6ac-453b-985d-1f9d53a01051.png">

### Step 2. Calculate semantic relevancy score
In this step, we calculated the semantic relevancy score between each small pieces of contexts we just got in step 1 and the questions. Then we can get the most relevant contexts and get answers from the contexts. To achieve this, we set a threshold for this step. If there are more than three pieces of contexts that have relevancy score higher than the threshold, we will use all the three contexts to provide final answers; otherwise we will use the most relevant context to get the answer.

<img width="562" alt="Screen Shot 2023-04-12 at 2 44 02 PM" src="https://user-images.githubusercontent.com/89158696/231567815-ad7da49c-b7f8-461f-9031-f665b134b7ce.png">

### Step 3. Model
First, `fine-tuning` we fine-tuned this model on [SQuAD2](https://rajpurkar.github.io/SQuAD-explorer/) dataset, so that the model can be used for question-answering tasks.

Second, `domain adaptation` we wrote down 500 pieces of annotations as training dataset to train the model and improve the model's performance on answering questions about airline industry companies' Corporate and Social Responsibility (CSR) reports.

Third, `testing` we got 77% accuracy to all kinds of answers (including questions that have text answers, questions that have numerical answers, and questions that don't have answers in the PDF document). Remarkably, we can accurately answer 93.4% questions that have numerical answers.

### Step 4. Product
We prepare to create a user interface in the future, which allows user to input the PDF and questions and can directly get the answers.

<img width="992" alt="Screen Shot 2023-04-12 at 3 12 13 PM" src="https://user-images.githubusercontent.com/89158696/231576579-aae52616-66c1-4dcb-ae05-0c3d3a92163b.png">


## Critical analysis
- This project revels part of the process of how ChatPDF works. The difference here is that ChatPDF has been trained with a lot of different types of pdf while this DitilBERT model was only retrained with Corporate and social responsibility (CSR) reports data.
- For the next step, we can increase the diversity of the training data so that the model will be able to process different types of pdf and answer all kinds of questions.

## Code demonstration
There are two files.

`model.ipynb` The main function we used

`extract_issue5.py` The function to extract information from PDF and save in json file.

## References

### Data

### Hugginface model

## Viedo link
