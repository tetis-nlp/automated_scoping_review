from grobid.tei import Parser
import json
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import pipeline


tei_files_path = "./data/grobid/"

tei_file_example = "./data/grobid/CC10 Aerosol Susceptibility of Influenza Virus to UV-C Light.pdf.tei.xml"

list_of_diseases = ["influenza", "chikungunya", "leptospirosi" ]

list_of_base_models =["roberta-base", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext".split("/")[-1], "FacebookAI/xlm-roberta-base".split("/")[-1]]
# list_of_base_models =["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "FacebookAI/xlm-roberta-base"]

sections_to_keep = ["abstract", "results", "discussion"]
chunk_size = 256

def chunk_relevant_sections(tei_file):
    chunks = []
    md = {
        "title": tei_file.split("/")[-1],
        "id": (tei_file.split("/")[-1]).split(" ")[0],
    }

    # load PDF into python dict
    with open(tei_file, "rb") as xml_file:
        xml_content = xml_file.read()
    parser = Parser(xml_content)
    article = parser.parse()
    article = json.loads(article.to_json())

    # work on abstract
    abstract = article["abstract"]
    for p, paragraph in enumerate(abstract["paragraphs"]):
        for i in range(0, len(paragraph["text"]), chunk_size):
            # print(f"{i} : {(i)} | {i + chunk_size -1}")
            chunk_with_md = md.copy()
            chunk_with_md["section"] = "abstract"
            chunk_with_md["paragraph_nb"] = p
            chunk_with_md["chunk_nb"] = i/chunk_size
            chunk_with_md["text"] = paragraph["text"][i:i+chunk_size-1]
            chunks.append(chunk_with_md)

    # work on usefull sections
    for s in article["sections"]:
        if (s["title"].lower() in sections_to_keep):
            for p, paragraph in enumerate(s["paragraphs"]):
                for i in range(0, len(paragraph["text"]), chunk_size):
                    # print(f"{i} : {(i)} | {i + chunk_size -1}")
                    chunk_with_md = md.copy()
                    chunk_with_md["section"] = s["title"]
                    chunk_with_md["paragraph_nb"] = p
                    chunk_with_md["chunk_nb"] = i/chunk_size
                    chunk_with_md["text"] = paragraph["text"][i:i+chunk_size-1]
                    chunks.append(chunk_with_md)
    
    return chunks

import os

for disease in list_of_diseases:
    tei_files_path_disease = tei_files_path + "/" + disease

    list_of_chunk = []
    for doc in os.listdir(tei_files_path_disease):
        list_of_chunk.extend(chunk_relevant_sections( os.path.join(tei_files_path_disease, doc)))
    df = pd.DataFrame(list_of_chunk)

    list_of_training_dataset = ["base", "gpt3-5-augmented"]

    # Function to check if a word is a stopword
    def is_stopword(word):
        # List of stopwords
        stopwords = ["the", "and", "is", "in", "to", "of", "a", "an", "al", "that", "for", "it", "with", "as", "are", "has", "by", "on", "was", "have", "we", "were", "at", "res", "one", "al.", "].", "or", "),", "et", "he"]
        return word.lower() in stopwords

    for pretrained_model in list_of_base_models:
        print(f"Work on: {pretrained_model}")
        for dataset in list_of_training_dataset:
            print(f"\t on: {dataset}")
            model_path = f"./models/mood_covariate_from_{pretrained_model}_on_{dataset}"
            # Load the model using the pipeline for Named Entity Recognition (NER)
            ner_classifier = pipeline("ner", model=model_path, aggregation_strategy="simple", device="cuda")

            def extract_covariates(text):
                res = ner_classifier(text)
                list_covariate = []
                for entity in res:
                    if entity["entity_group"] != '0':
                        if(len(entity["word"].lstrip()) > 1 ): # More than only one letter
                            if(is_stopword(entity["word"].lstrip()) == False): #remove stopwords
                                list_covariate.append(entity["word"].lstrip())
                return "" if len(list_covariate) == 0 else str(list_covariate)
            
            df[model_path.split("/")[-1]] = df["text"].apply(extract_covariates)

    df.to_csv(f"./data/whole_inference_{disease}.csv")