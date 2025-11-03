# Automated Updates for Scoping Reviews of Environmental Drivers of Human and Animal Diseases

This study aims to compare three NLP methods for extracting named entities with complex labels and very limited training data.

We compare:

- Fine-tuning BERT on NER classification
- Data augmentation with GPT-3.5 and fine-tuning BERT on both the original and data-augmented training datasets
- OpenAI (GPT-3.5 and GPT-4) with RAG based on the same training dataset

We trained our methods on an Influenza corpus and evaluated the ability of these approaches to generalize to other diseases (Leptospirosis and Chikungunya).

## Reproduce the Article

### 0. Download the Papers and Their Manual Annotations
1. Download the manual annotations:

- [Manual scoping review made by the MOOD project](https://doi.org/10.5281/zenodo.11241409). Manual annotation at document level with normalization of entities extracted
- Manual annotation at sentence level: *Work in progress*

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

### 4. Analysing differences between generalization among others diseases
As regards the generalization across diseases practical examples have been gathered in a sheet ([[link](analysing_results/AI_MODELS_COMPARISON.xlsx)]). Three articles were selected for each disease (CC22,CC17,CC12). A special focus was made on `Humidity` as covariate to explain the similarities and/or differences in impact among the diseases focusing  on similarities and contextualization of the terms used in scientific language in the text extracted by the models. As illustrated by the sheet, although you can find "Humidity" as a word in all the diseases, the related other words are more similar between influenza and leptospirosis than influenza and Chikungunya (words highlighted  in bold). 

--------------------
## Citing this work

If you use this project in your research, please cite the following article:

### BibTeX
```bibtex
@article{Decoupes2025,
  title   = {Automating updates for scoping reviews on the environmental drivers of human and animal diseases: a comparative analysis of AI methods},
  author  = {Decoupes, Rémy and Cataldo, Claudia and Busani, Luca and Roche, Mathieu and Teisseire, Maguelonne},
  journal = {Frontiers in Artificial Intelligence},
  year    = {2025},
  volume  = {8},
  pages   = {1526820},
  doi     = {10.3389/frai.2025.1526820},
  url     = {https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1526820/full}
}
```
--------------------
**Acknowledgement**:

This study was partially funded by EU grant 874850 MOOD. The contents of
this publication are the sole responsibility of the authors and do not necessarily reflect the views of the European
Commission

<a href="https://mood-h2020.eu/"><img src="https://mood-h2020.eu/wp-content/uploads/2020/10/logo_Mood_texte-dessous_CMJN_vecto-300x136.jpg" alt="mood"/></a> 
