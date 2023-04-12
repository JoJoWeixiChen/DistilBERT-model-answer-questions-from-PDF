#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 13:07:45 2023

@author: chenweixi
"""

#! pip install sentence_transformers
############################# EXTRACT JSON FROM PDF #################################
import logging
import os.path
import zipfile
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer

from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from transformers import AutoModelForQuestionAnswering

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def extract_table_from_pdf(path1, path2):
    try:
    # get base path.
        base_path = os.path.abspath(path1)

    # Initial setup, create credentials instance.
        credentials = Credentials.service_account_credentials_builder() \
                .from_file(base_path + "/pdfservices-api-credentials.json") \
                .build()

    # Create an ExecutionContext using credentials and create a new operation instance.
        execution_context = ExecutionContext.create(credentials)
        extract_pdf_operation = ExtractPDFOperation.create_new()
    
        for i in os.listdir(base_path + path2)[1:]:

        # Set operation input from a source file.
            source = FileRef.create_from_local_file(base_path + path2 + i)
            extract_pdf_operation.set_input(source)

        # Build ExtractPDF options and set them into the operation
            extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder() \
                    .with_element_to_extract(ExtractElementType.TEXT) \
                    .with_element_to_extract(ExtractElementType.TABLES) \
                    .build()
            extract_pdf_operation.set_options(extract_pdf_options)

        # Execute the operation.
            result: FileRef = extract_pdf_operation.execute(execution_context)

        # Save the result to the specified location.
            result.save_as(base_path + "/output/" + i[:-4] + '.zip')
    except (ServiceApiException, ServiceUsageException, SdkException):
        logging.exception("Exception encountered while executing operation")
 
    

############################# UNZIP .ZIP #################################
def unzip_document(path1):
    dir_name = path1 + '/output/'
    os.chdir(dir_name)
    for file in os.listdir():
    #print(file)# get the list of files
        if zipfile.is_zipfile(file): 
        #print(zipfile)# if it is a zipfile, extract it
            with zipfile.ZipFile(file) as item: # treat the file as a zip
                item.extractall()  # extract it in the working directory
                os.rename("structuredData.json", file[:-4] + ".json")

############################# EXTRACT TEXT INFORMATION #################################
def extract_text_from_json(file):
    with open(path1 + '/output/' + file, "r") as json_file:
        y = json.load(json_file)
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    out = pd.DataFrame(out.items())
    out.columns = ['Key', 'Content']
    df = out[out['Key'].str.endswith('Text')]
    return df
    
############################# CALCULATE RELEVANCY SCORE FOR ALL CONTEXTS #################################
def semantic_score(data, question, context_column):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    new_data = data.dropna(subset = [context_column])
    corpus = new_data[context_column].tolist()
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    queries = [question]
    top_k = min(3, len(corpus))
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 3 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        
        score_list = []
        context_list = []
        for score, idx in zip(top_results[0], top_results[1]):
            score_list.append(float(score))
            context_list.append(corpus[idx])
    
    result = {'context': context_list, 'score': score_list}
    result = pd.DataFrame(result)
    result = ' '.join(result['context'])
    return result

############################# DATA FORMAT TRANSFORM #################################
#def form_validation_data():
    

############################# CREATING THE MODEL #################################



   
# The base path
path1 = '/Users/chenweixi/Desktop/extract_table_from_pdf/adobe-dc-pdf-services-sdk-extract-python-samples'
# The path for PDF
path2 = '/resources/'

extract_table_from_pdf(path1, path2)
unzip_document(path1)
data = extract_text_from_json('aa.json')

a = extract_text_from_json('aa.json')
result = semantic_score(a, 'How many pilots are voluntering for project wingman', 'Content')

model_checkpoint = 'distilbert-base-uncased'
model = AutoModelForQuestionAnswering.from_pretrained('/Users/chenweixi/Desktop/my_model', local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

question = 'How many pilots are voluntering for project wingman'
context = result
inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)