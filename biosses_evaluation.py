from datasets import load_dataset, concatenate_datasets
from sklearn.metrics.pairwise import cosine_similarity


from save_load_models import load_model_sentencetransformer

multiMini = './Models/multilingual-MiniLM'
msMarcoMini = './Models/msmarco-MiniLM'
wikimedicalBioBert = './Models/wikimedical-BioBERT'
sPubMedBert = './Models/SPubMedBERT'

sPubMedBertFinetunedv1 = './Models/Finetunedv1/SPubMedBERT/end_model'
sPubMedBertFinetunedv2 = './Models/Finetunedv2/SPubMedBERT/end_model'
sPubMedBertFinetuneNfcorpus = './Models/Finetuned/SPubMedBERT-v1-nfcorpus'

pubMedBertV1Nfcorpus = './Models/Finetuned/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext-v1-nfcorpus'

model = load_model_sentencetransformer(sPubMedBertFinetunedv2)

biosses_ds = load_dataset("bigbio/biosses", "biosses_bigbio_pairs")

biosses_ds = concatenate_datasets([biosses_ds[split] for split in biosses_ds.keys()])

def scale_scores(scores):
    """
    Scale the resulting similarity scores to be between 0 and 4.
    This is necessary, as some models always have high scores and the dataset has label-scores between 0 and 4.

    Parameters:
    - scores (dict): {id:score} Pair-ID and corresponding calculated value.
    """

    # Find minimal and maximal scores
    min_score = min(scores.values())
    max_score = max(scores.values())

    if min_score == max_score:
        return scores
    
    # Scale all values between 0 and 4
    for key in scores.keys():
        scores[key] = 4 * (scores[key] - min_score) / (max_score - min_score)
    
    return scores


def mean_squared_error(dataset, score_dict):
    """
    Calculate the Mean Squared Error between true and predicted values.
    
    Parameters:
        dataset (dataset): Dataset with at least id and label fields.
        score_dict (dict): Dictionary with ids and corresponding predicted scores.
    """
    errors = []
    
    for e in dataset:
        if e['id'] in score_dict:
            errors.append((float(e['label']) - score_dict[e['id']]) ** 2)
        
    return sum(errors) / len(errors)


def pearson_correlation(dataset, score_dict):
    """
    Calculate the Pearson correlation coefficient between dataset and predicted values.
    
    Parameters:
        dataset (dataset): Dataset with at least id and label fields.
        score_dict (dict): Dictionary with ids and corresponding predicted scores.
    """

    n = len(dataset)

    dataset_labels = []
    pred_scores = []

    for e in dataset:
        if e['id'] in score_dict:
            dataset_labels.append(float(e['label']))
            pred_scores.append(score_dict[e['id']])

    sum_x = sum(dataset_labels)
    sum_y = sum(pred_scores)
    sum_xy = sum(xi * yi for xi, yi in zip(dataset_labels, pred_scores))
    sum_x_squared = sum(xi ** 2 for xi in dataset_labels)
    sum_y_squared = sum(yi ** 2 for yi in pred_scores)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x_squared - sum_x ** 2) * (n * sum_y_squared - sum_y ** 2)) ** 0.5
    
    if denominator == 0:
        return 0  # Avoid division by zero
    
    return numerator / denominator



scores = {}

# Calculate the cosine similarities between the texts
for sentence_pair in biosses_ds:
    text_1_vect = model.encode(sentence_pair['text_1'])
    text_2_vect = model.encode(sentence_pair['text_2'])
    scores[sentence_pair['id']] = cosine_similarity([text_1_vect] , [text_2_vect]).item()
#print(biosses_ds[:10])

# Scale the scores between 0 and 4 for the comparison
scaled_scores = scale_scores(scores)

# Evaluate the model based on the scaled scores
mse = mean_squared_error(dataset=biosses_ds, score_dict=scaled_scores)
print(f"MSE:     {mse:.4f}")

pearson = pearson_correlation(dataset=biosses_ds, score_dict=scaled_scores)
print(f"Pearson: {pearson:.4f}")


# sPubMedBert: 0.3195; 0.8827
#SPubMedBertv2: 0.9897; 0.8391
# sPubMedBertFinetuneNfcorpus: 0.6418; 0.8584

# pubMedBert: 1.1379; 0.8145