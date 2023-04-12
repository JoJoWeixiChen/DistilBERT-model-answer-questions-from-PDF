# DistillBERT-model-answer-questions-from-PDF

## Background
Every day, the financial industry analysts are tasked with synthesizing and summarizing huge number of documents or extracting important themes or metrics from these documents.  These days, ESG-oriented strategies have become more mainstream.  Here, ESG represents Environmental, Social and Governance.  Asset managers are looking for ways to assess ESG-related activities in their investment companies and to monitor their progress towards their goals. Among many ESG related documents, Corporate and social responsibility (CSR) reports are used by companies to communicate their CSR efforts and their impact on the environment and community. Though it is not required for a company to publish its CSR report annually, more than 90% of the companies in the S&P 500 Index have done so for 2019.  Within AB, a common problem within Fixed income responsibility investment team is to find certain ESG metrics to answer certain questions. 

**Project Objective**
Using QA model(s) to extract ESG-related answers from CSR Reports for Industry specific data (beginning with the Airlines Industry)

**Main Project Tasks**
- Annotation
  1. Become familiar with Sandford SQuAD dataset and annotate 1000 context-question-answer samples for Text QA model (This data will be used during weeks 1 - 4)
  2. Get familiar with datasets like SQA (Sequential Question Answering by Microsoft), WTQ (Wiki Table Questions by Stanford University), WikiSQL (by Salesforce) and annotate 1000 context-question-answer samples for Table QA model (This data will be used in weeks 5 -7)
    - Note that answers may come in different forms (not only from text portion or a table but also how the questions are answered.)
- Extract ESG-related metrics from texts using the Text QA model
  1. Preparation and set up (Week 1)
  2. Data preprocessing (Weeks 2 - 4)
  3. Text QA baseline result (Week 5)
  4. Text QA Finetuning (Weeks 6 - 8)
- Extract ESG-related metrics from tables using the Tabular QA model
  1. Preparation and Data preprocessing (Week 9)
  2. Tabular QA baseline result (Week 10)
  3. Tabular QA Finetuning (Weeks 11 - 13)

**Project Deliverables**
-	~1K annotations for Text QA and ~1k annotations for Tabular QA
-	Well-documented and fine-tuned QA model(s) for ESG that can be directly used by AllianceBernstein
-	Weekly project demos: Monday 1-2 pm
-	A technical report describing project specifics, e.g., documenting data pre-processing steps, how to use/select them for fine-tuning models, details of experiments conducted, necessary steps for reproducing projects, etc.
- Formal presentation (will be scheduled later)

## Model

## Pipeline

## Critical analysis

## Code demonstration

## References

### Data

### Hugginface model

## Viedo link
