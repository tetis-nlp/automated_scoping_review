import pandas as pd
import transformers
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

list_of_diseases = ["influenza", "chikungunya", "leptospirosi" ]

def is_substring(word, word_list):
    for other_word in word_list:
        if word != other_word and word in other_word:
            return True
    return False

def clean_and_flatten(list_of_lists):
    # print(f"type: {type(list_of_lists)}")
    flattened_list = []
    for sublist in list_of_lists:
        # print(f'sublist: {sublist}')
        try: #remove np.nan
            for element in ast.literal_eval(sublist):
                # print(f'element: {element}')
                flattened_list.append(element.lower())
        except:
            pass
    flattened_list = [word for word in flattened_list if not is_substring(word, flattened_list)]
    unique_values = list(set(flattened_list))
    return unique_values

for disease in list_of_diseases:
    if disease == "influenza":
        df_ground_truth = pd.read_excel('./data/corpus_AI_Luca/Data_Influenza/Influenza_DATA Extraction_MOOD_NO PIVOT.xlsx', "ENV_COV")
    elif disease == "chikungunya":
        df_ground_truth = pd.read_excel('./data/corpus_AI_Luca/Chikungunya data/Chikungunya_DATA Extraction_MOOD (1).xlsx', "ENV_COV")
    else:
        df_ground_truth = pd.read_excel('./data/corpus_AI_Luca/Leptospirosi Data/Leptospirosis_DATA Extraction_MOOD.xlsx', "ENV_COV")
    
    df_predicted = pd.read_csv(f"./src/2nd_annotation/data/whole_inference_{disease}.csv")

    gb = df_predicted.groupby("id")
    list_flatten = []
    for id in df_predicted["id"].unique():
        element_dict = {
            "id": id,
            'mood_covariate_from_roberta-base_on_base': clean_and_flatten(gb.get_group(id)['mood_covariate_from_roberta-base_on_base'].tolist()),
            'mood_covariate_from_roberta-base_on_gpt3-5-augmented': clean_and_flatten(gb.get_group(id)['mood_covariate_from_roberta-base_on_gpt3-5-augmented'].tolist()),
            'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_base': clean_and_flatten(gb.get_group(id)['mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_base'].tolist()),
            'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_gpt3-5-augmented': clean_and_flatten(gb.get_group(id)['mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_gpt3-5-augmented'].tolist()),
            'mood_covariate_from_xlm-roberta-base_on_base': clean_and_flatten(gb.get_group(id)['mood_covariate_from_xlm-roberta-base_on_base'].tolist()),
            'mood_covariate_from_xlm-roberta-base_on_gpt3-5-augmented': clean_and_flatten(gb.get_group(id)['mood_covariate_from_xlm-roberta-base_on_gpt3-5-augmented'].tolist()),
        }
        list_flatten.append(element_dict)

    df_predicted_flatten = pd.DataFrame(list_flatten)

    gb = df_ground_truth.groupby("ID")
    list_flatten_truth = []
    for id in df_ground_truth["ID"].unique():
        element_dict = {
            "id": id,
            'ground_truth': list(set(gb.get_group(id)['COVARIATE_env'].tolist())),
        }
        list_flatten_truth.append(element_dict)

    df_ground_truth_flatten = pd.DataFrame(list_flatten_truth)
    df_flatten = pd.merge(df_predicted_flatten, df_ground_truth_flatten, on="id", how="inner")

    method = ['mood_covariate_from_roberta-base_on_base',
            'mood_covariate_from_roberta-base_on_gpt3-5-augmented',
            'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_base',
            'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_gpt3-5-augmented',
            'mood_covariate_from_xlm-roberta-base_on_base',
            'mood_covariate_from_xlm-roberta-base_on_gpt3-5-augmented']

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    list_best_pairs = []
    for i, id in df_flatten.iterrows():
        print(f"{id['id']}")
        embeddings_gt = model.encode(id["ground_truth"])
        best_pair_tmp = []
        for m in method:
            embeddings_predicted = model.encode(id[m])
            try:
                similarity_matrix = cosine_similarity(embeddings_gt, embeddings_predicted)
                threshold = 0.7
                pairs = []
                print(f"Max score: {similarity_matrix.max()}")
                best_pair_tmp.append(similarity_matrix.max())
                for i in range(len(id["ground_truth"])):
                    for j in range(len(id[m])):
                        if similarity_matrix[i][j] > threshold:
                            pairs.append({
                                'embedding1': embeddings_gt[i],
                                'embedding2': embeddings_predicted[j],
                                'ground_thruth': id["ground_truth"][i],
                                'predicted': id[m][j],
                                'cosine_similarity': similarity_matrix[i][j]
                            })

                # Print the extracted pairs
                for pair in pairs:
                    print(f"  |-> method: {m} | {id['id']}: Pair: {pair['ground_thruth']} - {pair['predicted']}, Cosine Similarity: {pair['cosine_similarity']}")
            except:
                print("  |-> ERROR")
                best_pair_tmp.append(0)
        try:
            best_matching =  {
                "id": id['id'],
                method[0]: best_pair_tmp[0],
                method[1]: best_pair_tmp[1],
                method[2]: best_pair_tmp[2],
                method[3]: best_pair_tmp[3],
                method[4]: best_pair_tmp[4],
                method[5]: best_pair_tmp[5],
            }
            list_best_pairs.append(best_matching)
        except:
            print("error")

    df_best_pair = pd.DataFrame(list_best_pairs)

    df_best_pair.to_csv(f"./src/2nd_annotation/data/document_level_{disease}.csv")

    df_plot = df_best_pair.copy()
    df_plot = df_plot.drop(columns=['id'])
    method_mapping = {
        'mood_covariate_from_roberta-base_on_base': 'roberta',
        'mood_covariate_from_roberta-base_on_gpt3-5-augmented': 'roberta_aug',
        'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_base': 'biomedbert',
        'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_gpt3-5-augmented': 'biomedbert_aug',
        'mood_covariate_from_xlm-roberta-base_on_base': 'xml_roberta',
        'mood_covariate_from_xlm-roberta-base_on_gpt3-5-augmented': 'xml_roberta_aug'
    }
    df_plot.rename(columns=method_mapping, inplace=True)

    plt.figure(figsize=(10, 6))
    df_plot.boxplot(rot=45)
    # plt.title(disease)
    plt.ylabel('Percentage of covariates found')
    plt.xlabel('Method')
    plt.tight_layout()
    plt.savefig(f"./src/2nd_annotation/data/document_level_{disease}.png")

