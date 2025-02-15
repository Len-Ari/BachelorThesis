# Slightly adjusted code from https://github.com/beir-cellar/beir/blob/main/beir/
# need to !pip install beir
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import List, Dict

import logging
import numpy as np
import pathlib, os
import random

from save_load_models import load_model_sentencetransformer

multiMini = './Models/multilingual-MiniLM'
msMarcoMini = './Models/msmarco-MiniLM'
wikimedicalBioBert = './Models/wikimedical-BioBERT'
sPubMedBert = './Models/SPubMedBERT'

sPubMedBertFinetunedv1 = './Models/Finetunedv1/SPubMedBERT/end_model'
sPubMedBertFinetunedv2 = './Models/Finetunedv2/SPubMedBERT/end_model'
sPubMedBertFinetuneNfcorpus = './Models/Finetuned/SPubMedBERT-v1-nfcorpus'

pubMedBertV1Nfcorpus = './Models/Finetuned/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext-v1-nfcorpus'

model_path = sPubMedBertFinetuneNfcorpus
embedding_strategy = 'split'

# nfcorpus
# trec-covid
dataset = "nfcorpus"

class YourCustomModel:
    def __init__(self, model_path=None, embedding_type='combined', **kwargs):
        self.model = load_model_sentencetransformer(model_path, usecpu=False)
        self.embedding_type = embedding_type
        # self.model = SentenceTransformer(model_path)
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    # For eg ==> return np.asarray(self.model.encode(queries, batch_size=batch_size, **kwargs))
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        return np.asarray(self.model.encode(queries))
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    # For eg ==> sentences = [(doc["title"] + "  " + doc["text"]).strip() for doc in corpus]
    #        ==> return np.asarray(self.model.encode(sentences, batch_size=batch_size, **kwargs))
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        embeddings = []
        if self.embedding_type == 'title':
            titles = [(doc["title"]).strip() for doc in corpus]
            embeddings = np.asarray(self.model.encode(titles))
        elif self.embedding_type == 'abstract':
            texts = [(doc["text"]).strip() for doc in corpus]
            embeddings = np.asarray(self.model.encode(texts))
        elif self.embedding_type == 'combined':
            documents = [(doc["title"] + "  " + doc["text"]).strip() for doc in corpus]
            embeddings = np.asarray(self.model.encode(documents))

        return embeddings
    

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

if embedding_strategy == 'split':
    #### Embedd and calculate scores for title embeddings
    model = DRES(YourCustomModel(model_path=model_path, embedding_type='title'))

    retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" if you wish dot-product

    results_1 = retriever.retrieve(corpus, queries)

    # Do the same for abstract embeddings
    model = DRES(YourCustomModel(model_path=model_path, embedding_type='abstract'))

    retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" if you wish dot-product

    results_2 = retriever.retrieve(corpus, queries)

    results = {}

    for query_key in results_1.keys():
        results[query_key] = {}
        if query_key in results_2:
            for doc_key in set(results_1[query_key]) | set(results_2[query_key]):
                #results[query_key][doc_key] = -1
                if doc_key in results_1[query_key] and doc_key in results_2[query_key]:
                    results[query_key][doc_key] = max(results_1[query_key][doc_key], results_2[query_key][doc_key])
                elif doc_key in results_1[query_key]:
                    results[query_key][doc_key] = results_1[query_key][doc_key]
                else:
                    results[query_key][doc_key] = results_2[query_key][doc_key]

elif embedding_strategy == 'combined':
    model = DRES(YourCustomModel(model_path=model_path, embedding_type='combined'))

    retriever = EvaluateRetrieval(model, score_function="cos_sim") # "cos_sim" or "dot"

    results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)









# NDCG@10 / nfcorpus

# MultiMini         (combined, split) = (0.2345, 0.2599)
# MsMarcoMini       (combined, split) = (0.2512, 0.2610)
# WikimedicalBERT   (combined, split) = (0.0761, 0.0851)

# sPubMedBERT       (combined, split) = (0.3120, 0.3224)
# sPubMedBERTfinev1 (combined, split) = (0.2549, 0.2567)
# sPubMedBERTfinev2 (combined, split) = (0.2898, 0.2898)

######

# NDCG@10 / trec-cpvid

# MultiMini         (combined, split) =
# MsMarcoMini       (combined, split) =
# WikimedicalBERT   (combined, split) = (, 0.3024)

# sPubMedBERT       (combined, split) =
# sPubMedBERTfinev1 (combined, split) =
# sPubMedBERTfinev2 (combined, split) =