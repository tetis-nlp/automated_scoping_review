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
import seaborn as sns
from Levenshtein import distance as levenshtein_distance

list_of_diseases = ["leptospirosi", "chikungunya", "influenza" ]
# list_of_diseases = ["chikungunya", "leptospirosi"]

method = ['mood_covariate_from_roberta-base_on_base',
        'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_base',
        'mood_covariate_from_xlm-roberta-base_on_base',
        'mood_covariate_from_roberta-base_on_gpt3-5-augmented',
        'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_gpt3-5-augmented',
        'mood_covariate_from_xlm-roberta-base_on_gpt3-5-augmented',
        'GPT-3.5',
        'GPT-4',
        ]
method_mapping = {
    'mood_covariate_from_roberta-base_on_base': 'roberta',
    'mood_covariate_from_roberta-base_on_gpt3-5-augmented': 'roberta_hybride',
    'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_base': 'biomedbert',
    'mood_covariate_from_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_on_gpt3-5-augmented': 'biomedbert_hybride',
    'mood_covariate_from_xlm-roberta-base_on_base': 'xml_roberta',
    'mood_covariate_from_xlm-roberta-base_on_gpt3-5-augmented': 'xml_roberta_hybride',
    'GPT-3.5': 'gpt-3.5',
    'GPT-4': 'gpt-4'
}

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

def split_and_clean(concepts_list):
    unique_concepts = set()  # Utilisation d'un set pour éviter les doublons

    for item in concepts_list:
        if isinstance(item, str):  # Vérifier que l'élément est bien une chaîne
            parts = item.replace("_x000D_", "").replace("\n", " ").split(";")  # Supprime "\n" et split sur ";"
            for part in parts:
                cleaned_part = part.strip().lower()  # Nettoyage des espaces et mise en minuscule
                if cleaned_part:
                    unique_concepts.add(cleaned_part)
    return list(unique_concepts)

from Levenshtein import distance as levenshtein_distance
import pandas as pd
import numpy as np

def compute_best_pairs_cosine(df_flatten, threshold=0.7):
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
                method[6]: best_pair_tmp[6],
                method[7]: best_pair_tmp[7],
            }
            list_best_pairs.append(best_matching)
        except:
            print("error")
    return pd.DataFrame(list_best_pairs)

def compute_best_pairs_Levenshtein(df_flatten, threshold=0.7):
    list_best_pairs = []

    for _, row in df_flatten.iterrows():
        print(f"{row['id']}")  # Print current ID
        best_pair_tmp = []  # Store best similarity scores for each method

        for m in method:
            try:
                gt_terms = row["ground_truth"]
                pred_terms = split_and_clean(row[m])  # Clean and split predictions

                similarity_matrix = np.zeros((len(gt_terms), len(pred_terms)))

                # Compute Levenshtein similarity (normalized)
                for i, gt_term in enumerate(gt_terms):
                    for j, pred_term in enumerate(pred_terms):
                        max_length = max(len(gt_term), len(pred_term)) or 1  # Avoid division by zero
                        similarity_matrix[i, j] = 1 - (levenshtein_distance(gt_term, pred_term) / max_length)

                max_score = similarity_matrix.max() if similarity_matrix.size > 0 else 0
                print(f"Max score: {max_score}")
                best_pair_tmp.append(max_score)

                pairs = []

                # Extract matching pairs above the threshold
                for i in range(len(gt_terms)):
                    for j in range(len(pred_terms)):
                        if similarity_matrix[i, j] > threshold:
                            pairs.append({
                                'ground_truth': gt_terms[i],
                                'predicted': pred_terms[j],
                                'levenshtein_similarity': similarity_matrix[i, j]
                            })

                # Print extracted pairs
                for pair in pairs:
                    print(f"  |-> method: {m} | {row['id']}: Pair: {pair['ground_truth']} - {pair['predicted']}, Levenshtein Similarity: {pair['levenshtein_similarity']}")

            except Exception as e:
                print(f"  |-> ERROR: {e}")
                best_pair_tmp.append(0)  # Append zero in case of error

        # Store best similarity scores for each method
        try:
            best_matching = {
                "id": row['id'],
                **{method[idx]: best_pair_tmp[idx] for idx in range(len(method))}
            }
            list_best_pairs.append(best_matching)
        except Exception as e:
            print(f"error: {e}")

    return pd.DataFrame(list_best_pairs)


for disease in list_of_diseases:
    if disease == "influenza":
        df_ground_truth = pd.read_excel('./data/corpus_AI_Luca/Data_Influenza/Influenza_DATA Extraction_MOOD_NO PIVOT.xlsx', "ENV_COV")
    elif disease == "chikungunya":
        df_ground_truth = pd.read_excel('./data/corpus_AI_Luca/Chikungunya data/Chikungunya_DATA Extraction_MOOD (1).xlsx', "ENV_COV")
    else:
        df_ground_truth = pd.read_excel('./data/corpus_AI_Luca/Leptospirosi Data/Leptospirosis_DATA Extraction_MOOD.xlsx', "ENV_COV")
    
    df_predicted = pd.read_csv(f"./src/2nd_annotation/data/whole_inference_{disease}.csv")
    df_predicted_llm = pd.read_csv(f"./src/2nd_annotation/data/whole_inference_llm_{disease}.csv")
    df_predicted = pd.concat([df_predicted, df_predicted_llm], ignore_index=True)

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
            'GPT-3.5': clean_and_flatten(gb.get_group(id)['GPT-3.5'].tolist()),
            'GPT-4': clean_and_flatten(gb.get_group(id)['GPT-4'].tolist()),
        }
        list_flatten.append(element_dict)

    df_predicted_flatten = pd.DataFrame(list_flatten)

    gb = df_ground_truth.groupby("ID")
    list_flatten_truth = []
    for id in df_ground_truth["ID"].unique():
        element_dict = {
            "id": id,
            # 'ground_truth': list(set(gb.get_group(id)['COVARIATE_env'].tolist())),
            'ground_truth': split_and_clean(list(set(gb.get_group(id)['COVARIATE_env'].tolist()))),
        }
        list_flatten_truth.append(element_dict)

    df_ground_truth_flatten = pd.DataFrame(list_flatten_truth)
    df_flatten = pd.merge(df_predicted_flatten, df_ground_truth_flatten, on="id", how="inner")
    df_flatten.to_csv(f"./src/2nd_annotation/data/review_all_predicted_document_level_{disease}.csv")


    df_best_pair = compute_best_pairs_cosine(df_flatten)
    df_best_pair_Levenshtein = compute_best_pairs_Levenshtein(df_flatten)

    df_best_pair.to_csv(f"./src/2nd_annotation/data/document_level_{disease}.csv")
    df_best_pair_Levenshtein.to_csv(f"./src/2nd_annotation/data/document_level_Levenshtein_{disease}.csv")

    df_plot = df_best_pair.copy()
    df_plot = df_plot.drop(columns=['id'])
    df_plot.rename(columns=method_mapping, inplace=True)

    plt.figure(figsize=(12, 6))

    
    df_plot.boxplot(rot=45)
    # plt.title(disease)
    plt.ylabel('Distribution of cosine similarity between ground truth and predicted')
    # plt.xlabel('Method')
    plt.tight_layout()
    plt.savefig(f"./src/2nd_annotation/data/document_level_{disease}.png")

# Cosine sim boxplot consolidated
df_disease1 = pd.read_csv(f"./src/2nd_annotation/data/document_level_influenza.csv")
df_disease2 = pd.read_csv(f"./src/2nd_annotation/data/document_level_leptospirosi.csv", index_col=0)
df_disease3 = pd.read_csv(f"./src/2nd_annotation/data/document_level_chikungunya.csv", index_col=0)

df_disease1['Disease'] = "influenza"
df_disease2['Disease'] = "leptospirosi"
df_disease3['Disease'] = "chikungunya"

df_combined = pd.concat([df_disease1, df_disease2, df_disease3])
df_melted = df_combined.melt(id_vars=['id', 'Disease'], var_name='Method', value_name='Value')
df_melted['Method'] = df_melted['Method'].map(method_mapping)
plt.figure(figsize=(12, 8))
sns.boxplot(x='Method', y='Value', hue='Disease', data=df_melted)
# plt.xlabel('Method')
plt.ylabel('cosine similarity value')
plt.xticks(rotation=45)
plt.legend(title='Disease')
plt.tight_layout()

plt.savefig(f"./src/2nd_annotation/data/document_level_all_disease.png")


# Levenshtein boxplot consolidated
df_disease1 = pd.read_csv(f"./src/2nd_annotation/data/document_level_Levenshtein_influenza.csv")
df_disease2 = pd.read_csv(f"./src/2nd_annotation/data/document_level_Levenshtein_leptospirosi.csv", index_col=0)
df_disease3 = pd.read_csv(f"./src/2nd_annotation/data/document_level_Levenshtein_chikungunya.csv", index_col=0)

df_disease1['Disease'] = "influenza"
df_disease2['Disease'] = "leptospirosi"
df_disease3['Disease'] = "chikungunya"

df_combined = pd.concat([df_disease1, df_disease2, df_disease3])
df_melted = df_combined.melt(id_vars=['id', 'Disease'], var_name='Method', value_name='Value')
df_melted['Method'] = df_melted['Method'].map(method_mapping)
plt.figure(figsize=(12, 8))
sns.boxplot(x='Method', y='Value', hue='Disease', data=df_melted)
# plt.xlabel('Method')
plt.ylabel('Levenshtein value')
plt.xticks(rotation=45)
plt.legend(title='Disease')
plt.tight_layout()

plt.savefig(f"./src/2nd_annotation/data/document_level_Levenshtein_all_disease.png")