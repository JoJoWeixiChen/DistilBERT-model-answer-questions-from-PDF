# DistilBERT-model-answer-questions-from-PDF

## Background
Nowadays, with the help of ChatPDF, we are able to extract information from large PDF documents easily. However, despite its effectiveness, the inner workings of this tool remain a mystery to most. Therefore, the purpose of this project is to shed light on the workings of ChatPDF, specifically in relation to its application in answering questions from airline industry companies' Corporate and Social Responsibility (CSR) reports.

### Problem
Extracting data from PDF documents was a time-consuming and tedious process, especially when the PDF documents are extremlly long. This often requires manual entry or the use of specialized software that may not always be accurate or reliable. However, with the application like ChatPDF and our model, users can get accurate answer from the PDF documents quickly.

### Project Objective
Build a model pipeline based on [DistilBERT-base-uncased model](https://huggingface.co/distilbert-base-uncased) that can understand text-based questions and extract information from PDF documents and answer questions accurately. Other than texts, there are often tables and graphs in PDF documents. This project is only focusing on extracting answers from the text, not from graphs and tables.

## Data
The data we used is the airline industry companies' Corporate and Social Responsibility (CSR) reports, including Delta, Jetblue, American Airlines, etc. We manually collected the training data online and annotated them to use for training the model. The title here is the keyword of the questions. The Context is the paragraph from the pdf document that contains the answer for the question. The questions is self-defined based on the context as well as the answers. The data size is relatively small, because it takes a lot of time and work to manually collect the data.

<img width="1310" alt="Screen Shot 2023-04-12 at 1 34 26 PM" src="https://user-images.githubusercontent.com/89158696/231579346-638de1fd-d5c0-4d83-9133-ee6b7bda55d2.png">


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
In this step, we used the sentence transformer model [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) to calculate the semantic relevancy score between each small pieces of contexts we just got in step 1 and the questions. The all-MiniLM-L6-v2 model can map sentences & paragraphs to a 384 dimensional dense vector space to calculate their semantic relevancy score. 

After getting the relevancy score between all context pieces and the questions, we set a threshold to select the potential contexts that may include the answers. 

If there are more than three pieces of contexts that have relevancy score higher than the threshold, we will use the contexts with top three relevancy score for predicting final answer; 

If there is just one piece of context that has relevancy score higher than the threshold, we will use the most relevant context to predict the answer.

Otherwise, we will return a feedback as 'No answer exists'.

<img width="562" alt="Screen Shot 2023-04-12 at 2 44 02 PM" src="https://user-images.githubusercontent.com/89158696/231567815-ad7da49c-b7f8-461f-9031-f665b134b7ce.png">

### Step 3. Model
First, `fine-tuning` we fine-tuned this model on [SQuAD2](https://rajpurkar.github.io/SQuAD-explorer/) dataset, so that the model can be used for question-answering tasks.

Second, `domain adaptation` we wrote down 500 pieces of annotations as training dataset to train the model and improve the model's performance on answering questions about airline industry companies' Corporate and Social Responsibility (CSR) reports.

Third, `testing` we got 77% accuracy to answer all kinds of questions (including questions that have text answers, questions that have numerical answers, and questions that don't have answers in the PDF document). Remarkably, we can accurately answer 93.4% questions that have numerical answers.

### Step 4. Product
We prepare to create a user interface in the future, which allows user to input the PDF and questions and can directly get the answers.

<img width="992" alt="Screen Shot 2023-04-12 at 3 12 13 PM" src="https://user-images.githubusercontent.com/89158696/231576579-aae52616-66c1-4dcb-ae05-0c3d3a92163b.png">


## Critical thinking
- This project revels part of the process of how ChatPDF works. The difference here is that ChatPDF has been trained with a lot of different types of pdf while this DitilBERT model was only retrained with Corporate and social responsibility (CSR) reports data.
- For the next step, we can increase the diversity of the training data so that the model will be able to process different types of pdf and answer all kinds of questions.

## Code
There are two files.

`model.ipynb` The main function we used

`extract_issue5.py` The function to extract information from PDF and save in json file.

## References

### Training Data
[Train data](https://github.com/JoJoWeixiChen/DistilBERT-model-answer-questions-from-PDF/blob/main/train_data.csv)

### PDF Data
1.	JetBlue [jb-2019-2020-esg-report.pdf (jetblue.com)](http://investor.jetblue.com/~/media/Files/J/Jetblue-IR-V2/Annual-Reports/jb-2019-2020-esg-report.pdf)
2.	Southwest https://www.southwest.com/assets/pdfs/communications/one-reports/2021_Southwest_One_Report_2022-04-22.pdf
3.	Citizen [2021-corporate_responsibility.pdf (citizensbank.com)](https://www.citizensbank.com/assets/pdf/2021-corporate_responsibility.pdf)
4.	[Delta ESG Report 2021 (delta.com)](https://esghub.delta.com/esg-report-2021)
5.	Ryanair  https://corporate.ryanair.com/sustainability-report-2021/ 
6.	AA [American Airlines - 2021 ESG Report](https://www.americanairlines.it/content/images/customer-service/about-us/corporate-governance/esg/aag-esg-report-2021.pdf)
7.	Cathay Pacific [Cathay_Pacific_SDR_2021_EN_Final.pdf (cathaypacific.com)](https://sustainability.cathaypacific.com/wp-content/uploads/2022/05/Cathay_Pacific_SDR_2021_EN_Final.pdf)
8.	Alaska Airline: https://sitecore-prod-cd-westus2.azurewebsites.net/-/media/5C9AD40027274CCA962DDAF60F7EA2E4 
9.	United Airline https://crreport.united.com/data/environment 
10.	LH [LH-Factsheet-Sustainability-2021.pdf (lufthansagroup.com)](https://www.lufthansagroup.com/media/downloads/en/responsibility/LH-Factsheet-Sustainability-2021.pdf)
11.	Qantas https://investor.qantas.com/FormBuilder/_Resource/_module/doLLG5ufYkCyEPjF1tpgyw/file/annual-reports/QAN_2022_Sustainability_Report.pdf 
12.	International Airline https://www.iairgroup.com/~/media/Files/I/IAG/documents/sustainability/sustainability-report-2021.pdf

### Hugginface model
DistilBERT-base-uncased model: https://huggingface.co/distilbert-base-uncased

## Viedo link

- [Project Overview Video Link](https://github.com/JoJoWeixiChen/DistilBERT-model-answer-questions-from-PDF/tree/main/Video%20Link) 

# Thanks
[Anubha](https://github.com/Anubha101)
[Jingting](https://github.com/Jingting723)
