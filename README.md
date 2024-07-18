# Automated Updates for Scoping Reviews of Environmental Drivers of Human and Animal Diseases


## Reproduce the Article

### 0. Download the Papers and Their Manual Annotations
1. Download the manual annotations:

- [Manual scoping review made by the MOOD project](https://doi.org/10.5281/zenodo.11241409)

2. Download the papers used for the manual annotations:
   - Download all the papers mentioned in the manual annotation using their DOI.

3. Convert PDFs into TEI:
   - We suggest using GROBID through its [HuggingFace space](https://huggingface.co/spaces/kermitt2/grobid).

### 1. Generate SpaCy-like Annotations
The two methods described below can be run using this notebook: [generate_annotation.ipynb](src/1_generate_annotation/generate_annotation.ipynb).

*Work in progress: the notebook needs to be adapted to the data from the Zenodo repository.*

**From Manual Annotations**:

The manual annotations are at the document level. To fine-tune BERT-like pre-trained models, we need to generate a SpaCy annotation schema.

**From Data Augmentation Using GPT-3.5**:

Use GPT-3.5 to create synthetic data from the manual annotations.

### 2. Train BERT-like Models and Create the RAG Process for LLMs

**Train BERT-like Models**:

Train 3 models:

- roberta-base
- microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
- FacebookAI/xlm-roberta-base

On two datasets:

- From the manual annotations
- From the manual annotations + synthetic data

All these 6 trainings can be done using this notebook: [train_models.ipynb](src/2_train_models/train_models.ipynb).

*Work in progress: the path to the training dataset needs to be adapted to the current environment.*

Then infer with the models trained on the whole datasets (the 3 diseases), using this script: [full_article_inference.py](src/3_infer/full_article_inference.py).

**RAG Process for LLMs**:

Create a RAG database (FAISS) and a Langchain pipeline for:

- GPT-3.5
- GPT-4

Using this notebook: [RAG.ipynb](src/2_train_models/RAG.ipynb).

### 3. Evaluate the Results
Compare cosine similarity between pairs (annotation/prediction). Extract only the best match for each article (even if some articles have several covariates annotated).

Run this script: [Evaluate_at_document_level.py](src/Evaluate_at_document_level.py).

--------------------
**Acknowledgement**:

This study was partially funded by EU grant 874850 MOOD. The contents of
this publication are the sole responsibility of the authors and do not necessarily reflect the views of the European
Commission

<a href="https://mood-h2020.eu/"><img src="https://mood-h2020.eu/wp-content/uploads/2020/10/logo_Mood_texte-dessous_CMJN_vecto-300x136.jpg" alt="mood"/></a> 
