# Any function needed to do a vector search or data extraction

import os
import json
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import torch



def add_embeddings_to_nparray_from_dir(dir_path, article_filter=[], use_pmc=False):
    '''
    Reads all jsonl files in the specified dir_path and saves the embeddings and corresponding filenames each in a numpy array.
    This then gets return as a tupel. 

    Parameters:
    - dir_path (String): A path to a directory containing jsonl files directly in it or any subdirectory.
    - article_filter (Array, optional): An Array of PubMed Ids and/or PubMedCentral Ids of documents to ignore when loading into memory.
    - use_pmc (Boolean, optional): If true return PMC-ID as reference, if False return Filename.
    
    Returns:
    - ndarray: A numpy array containing all embeddings extracted from the jsonl files.
    - ndarray (String): A numpy array containing filenames correspoding to the embeddings extracted by index.
    '''

    ret_list = []
    idx_ref_list = []

    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_file():
                # Check if file is json
                _, extension = os.path.splitext(entry.path)
                if extension == '.jsonl':
                    # If the entry is a jsonl-file we can extract all embeddings
                    # Add all embeddings to a list and convert to nparray at the end
                    # Read in Jsonl file
                    try:
                        file_in = open(entry.path, 'r', encoding='utf-8')
                    except OSError:
                        print ("Could not open/read file: ", entry.path)
                        sys.exit()

                    with file_in:
                        for line in file_in:
                            article_data = json.loads(line)
                            # Get pmid and pmc id
                            pmid = ''
                            pmcid = ''
                            if 'pmid' in article_data['metadata']['article_ids']:
                                pmid = article_data['metadata']['article_ids']['pmid']
                            if 'pmc' in article_data['metadata']['article_ids']:
                                pmcid = article_data['metadata']['article_ids']['pmc']
                            elif use_pmc:
                                continue
                            # Ignore file if either pmid or pmc in article_filter
                            if pmid in article_filter or pmcid in article_filter:
                                continue

                            # Article is valid and all components are loaded into array
                            if len(article_data['embeddings']['title']) != 0:
                                ret_list.append(article_data['embeddings']['title'])
                                idx_ref_list.extend(["Title-"+pmcid])
                            if len(article_data['embeddings']['abstract']) != 0:
                                ret_list.append(article_data['embeddings']['abstract'])
                                idx_ref_list.extend(["Abstract-"+pmcid])
                            if len(article_data['embeddings']['article']) != 0:
                                ret_list.append(article_data['embeddings']['article'])
                                idx_ref_list.extend(["Article-"+pmcid])
                            #idx_ref_list.extend(["Title-"+article_data['filename'], "Abstract-"+article_data['filename'], "Article-"+article_data['filename']])
                    file_in.close()

            elif entry.is_dir():
                # Recursivly call this function until dir_path is a file...
                ret_l, ref_l = add_embeddings_to_nparray_from_dir(entry.path, article_filter)
                ret_list.extend(ret_l)
                idx_ref_list.extend(ref_l)

        # Added all embeddings to ret_list and filenames to ref_list
        return np.array(ret_list), np.array(idx_ref_list, dtype=object)


def calc_all_embedding_similarity(embedding_list, scaled=False):
    '''
    For each datapoint: calculate the cosine similarity to all other points and return result.
    '''
    sims = cosine_similarity(np.array(embedding_list).squeeze())
    if scaled:
        min_sim, max_sim = find_min_max_similarity(embedding_list)
        sims = (sims - min_sim)/(max_sim-min_sim)
    return sims


def find_min_max_similarity(embedding_list):
    '''
    For each datapoint: calculate the cosine similarity to all other points and subsequently return the minimum and maximum similarity.
    Return complete average at the end.
    '''
    sims = cosine_similarity(np.array(embedding_list).squeeze())
    return sims.min(), sims.max()


import helper_methods as hm
def query_articles(queries, embedding_list, ref_list, model, model_type='sentence', tokenizer=None, n=5, print_results=True, fit_to_length=None):
    '''
    Calculate the closest embeddings using cosine-similarity for a list of queries.
    Returns the results in sorted order.
    If n==0 return all results unordered.
    
    Parameters:
    - queries (list<String>): A list of Query-Strings for the model to embedd.
    - embedding_list (ndarray): A numpy array consisting of the embeddings for the whole dataset.
    - ref_list (ndarray<String>): A numpy array consisting of the filenames corresponding to the embedding_list by index.
    - model (sentenceTRansformer): A sentence transformer model to comute embeddings for queries.
    - model_type (String, optional): Standard value='sentence'. A string representing the model type so that the correct function can be called.
        ('sentence', 'huggingface')    
    - tokenizer (for huggingface models): A Tokenizer to preprocess input for huggingface transformer models.
    - n (int, optional): Standard value = 5. Amount of search results to return. 0 <= n <= len(embedding_list)
    - print_results (boolean, optional): Standard value = True. Wether or not to print the filenames of the results
    - fit_to_length(int, optional): Loop input until certain length

    Returns:
    - query_res_ref (List<List>): The top results for each query in a List indexed by the query index.
    - query_res_cosine (List<List>): The cosine values corresponding to the references in query_res_ref.
    - query_embeddings (ndarray): Array of embeddings for each query
    '''

    top_sim_cosine, top_sim_ref = [], []

    n = min(max(0, n), len(embedding_list))

    if(fit_to_length != None):
        for idx, q in enumerate(queries):
            q_new = (q + ' . ') * (fit_to_length//(len(q.split())+1))
            queries[idx] = q_new

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if model_type == 'sentence':
        query_embeddings = model.encode(queries)
    elif model_type == 'huggingface':
        query_embeddings = hm.huggingface_transformer_encode(queries, model, tokenizer, device=device)
    similarities = cosine_similarity(query_embeddings, np.array(embedding_list).squeeze())
    if n==0:
        # Return all similariy scores without reordering
        return ref_list, similarities, query_embeddings
    top_sim_idx = np.argpartition(similarities, -n, axis=1)
    for i, idx_list in enumerate(top_sim_idx):
        top_sim_cosine.append(similarities[i][idx_list][-n:])
        top_sim_ref.append(ref_list[idx_list][-n:])
        order_idx = np.argsort(top_sim_cosine[i])[::-1]
        top_sim_ref[i] = top_sim_ref[i][order_idx]
        top_sim_cosine[i] = top_sim_cosine[i][order_idx]
        if print_results:
            print(f"The top results for the Query: <<{queries[i]}>> were:\n{top_sim_ref[i]}\nWith the cosine:\n", top_sim_cosine[i])
    
    return top_sim_ref, top_sim_cosine, query_embeddings


def print_article(article_dict, print_filename=True, print_abstract=True, print_article_content=False, print_metadata=False):
    '''
    Format Articles s.t. they are printed nicely in the console.

    Parameters:
    - article_dict (dict): A dictionary containing all information about the article. The form used must be the same as used usually...
    - print_filename (boolean, optional): Standard value = True
    - print_abstract (boolean, optional): Standard value = True
    - print_article_content (boolean, otional): Standard value = False
    - print_metadata (boolean, optional): Standard value = False

    Returns:
    - None
    '''
    print("\n Filename: ", article_dict['filename'])
    if print_metadata:
        print(article_dict['metadata'])
    else:
        print("\nTitle:\n", article_dict['metadata']['article_title'])
    
    if print_abstract:
        print("\nAbstract:\n", article_dict['text_content']['abstract'])
    if print_article_content:
        print("\nArticle:\n", article_dict['text_content']['article'])
    print("\n--------------------------------------------------------\n")


def get_articles(article_filenames, dir_path, print_results=True):
    '''
    Print all articles specified in a list of filenames. (Only 1 dimensional lists)
    Specify file location in dir_path.

    Parameters:
    - article_filenames (Array): A list/array of the filenames corresponding to the articles to be extracted.
    - dir_path (String): A path to a directory containing jsonl files directly in it or any subdirectory.
    - print_results (boolean, optional): Standard value = True. Wether or not to print the articles or just return.

    Returns:
    - articles (List<dict>): A List of dictionaries containing all information about the article in order as given in article_filenames.
    '''

    article_buff = [None] * len(article_filenames)
    article_filenames = np.array(article_filenames).squeeze()

    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_file():
                # Check if file is json
                _, extension = os.path.splitext(entry.path)
                if extension == '.jsonl':
                    # If the entry is a jsonl-file we can search for filenames
                    # Read in Jsonl file
                    try:
                        file_in = open(entry.path, 'r', encoding='utf-8')
                    except OSError:
                        print ("Could not open/read file: ", entry.path)
                        sys.exit()

                    with file_in:
                        for line in file_in:
                            article_data = json.loads(line)
                            if 'pmc' in article_data['metadata']['article_ids']:
                                pmcid = article_data['metadata']['article_ids']['pmc']
                            elif use_pmc:
                                continue
                            if len(article_filenames[0].split("-")) > 1:
                                tmp = [a.split("-")[1] for a in article_filenames]
                            else:
                                tmp = list(article_filenames)
                            if article_data['filename'] in tmp:
                                idx = tmp.index(article_data['filename'])
                                article_buff[idx] = article_data
                            elif pmcid in tmp:
                                idx = tmp.index(pmcid)
                                article_buff[idx] = article_data
                    file_in.close()
            elif entry.is_dir():
                # Recursivly call this function until dir_path is a file.
                article_buff_rec = get_articles(article_filenames, entry.path, print_results=False)
                for idx, val in enumerate(article_buff):
                    if val == None:
                        article_buff[idx] = article_buff_rec[idx]
    
    if print_results:
        for article in article_buff:
            if article != None:
                print_article(article) 

    return article_buff


def filter_embedding_types(embeddings, ref_list, embedding_types=['Abstract']):
    '''
    Return edited embeddings and ref_list only containing embeddings of origin embedding_types.
    '''
    ret_embeddings = []
    ret_ref_list = []
    embeddings = np.squeeze(embeddings)
    ref_list = np.squeeze(ref_list)
    for idx, ref in enumerate(ref_list):
        embed_type = ref.split("-")[0]
        if embed_type in embedding_types:
            ret_embeddings.append(embeddings[idx])
            ret_ref_list.append(ref)
    return ret_embeddings, ret_ref_list


def add_embeddings_and_text_to_nparray_from_dir(dir_path, article_filter=[]):
    '''
    Reads all jsonl files in the specified dir_path and saves the embeddings and corresponding filenames each in a numpy array.
    This then gets return as a tupel. 

    Parameters:
    - dir_path (String): A path to a directory containing jsonl files directly in it or any subdirectory.
    - article_filter (Array, optional): An Array of PubMed Ids and/or PubMedCentral Ids of documents to ignore when loading into memory

    Returns:
    - ndarray: A numpy array containing all embeddings extracted from the jsonl files.
    - ndarray (String): A numpy array containing filenames correspoding to the embeddings extracted by index.
    '''

    ret_embed_list = []
    ret_text_list = []
    idx_ref_list = []

    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_file():
                # Check if file is json
                _, extension = os.path.splitext(entry.path)
                if extension == '.jsonl':
                    # If the entry is a jsonl-file we can extract all embeddings
                    # Add all embeddings to a list and convert to nparray at the end
                    # Read in Jsonl file
                    try:
                        file_in = open(entry.path, 'r', encoding='utf-8')
                    except OSError:
                        print ("Could not open/read file: ", entry.path)
                        sys.exit()

                    with file_in:
                        for line in file_in:
                            article_data = json.loads(line)
                            # Get pmid and pmc id
                            if 'pmid' in article_data['metadata']['article_ids']:
                                pmid = article_data['metadata']['article_ids']['pmid']
                            if 'pmc' in article_data['metadata']['article_ids']:
                                pmcid = article_data['metadata']['article_ids']['pmc']
                            # Ignore file if either pmid or pmc in article_filter
                            if pmid in article_filter or pmcid in article_filter:
                                continue

                            # Article is valid and all components are loaded into array
                            if len(article_data['embeddings']['title']) != 0:
                                ret_embed_list.append(article_data['embeddings']['title'])
                                ret_text_list.append(article_data['metadata']['article_title'])
                                idx_ref_list.extend(["Title-"+article_data['filename']])
                            if len(article_data['embeddings']['abstract']) != 0:
                                ret_embed_list.append(article_data['embeddings']['abstract'])
                                ret_text_list.append(article_data['text_content']['abstract'])
                                idx_ref_list.extend(["Abstract-"+article_data['filename']])
                            if len(article_data['embeddings']['article']) != 0:
                                ret_embed_list.append(article_data['embeddings']['article'])
                                ret_text_list.append(article_data['text_content']['article'])
                                idx_ref_list.extend(["Article-"+article_data['filename']])
                            #idx_ref_list.extend(["Title-"+article_data['filename'], "Abstract-"+article_data['filename'], "Article-"+article_data['filename']])
                    file_in.close()

            elif entry.is_dir():
                # Recursivly call this function until dir_path is a file...
                ret_e_l, ret_t_l, ref_l = add_embeddings_and_text_to_nparray_from_dir(entry.path, article_filter)
                ret_embed_list.extend(ret_e_l)
                ret_text_list.extend(ret_t_l)
                idx_ref_list.extend(ref_l)

        # Added all embeddings to ret_list and filenames to ref_list
        return np.array(ret_embed_list), np.array(ret_text_list), np.array(idx_ref_list, dtype=object)



def get_title_abstract_from_dir(dir_path, article_filter=[]):
    '''
    Returns all text passages...
    '''

    ret_text_list = []
    idx_ref_list = []

    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_file():
                # Check if file is json
                _, extension = os.path.splitext(entry.path)
                if extension == '.jsonl':
                    # If the entry is a jsonl-file we can extract all embeddings
                    # Add all embeddings to a list and convert to nparray at the end
                    # Read in Jsonl file
                    try:
                        file_in = open(entry.path, 'r', encoding='utf-8')
                    except OSError:
                        print ("Could not open/read file: ", entry.path)
                        sys.exit()

                    with file_in:
                        for line in file_in:
                            article_data = json.loads(line)
                            # Get pmid and pmc id
                            if 'pmid' in article_data['metadata']['article_ids']:
                                pmid = article_data['metadata']['article_ids']['pmid']
                            if 'pmc' in article_data['metadata']['article_ids']:
                                pmcid = article_data['metadata']['article_ids']['pmc']
                            else:
                                continue
                            # Ignore file if either pmid or pmc in article_filter
                            if pmid in article_filter or pmcid in article_filter:
                                continue

                            # Article is valid and all components are loaded into array
                            try:
                                title = article_data['metadata']['article_title'] or ""
                                abstract = article_data['text_content']['abstract'] or ""
                                ret_text_list.append(title)
                                idx_ref_list.extend(["Title-"+article_data['metadata']['article_ids']['pmc']])
                                ret_text_list.append(abstract)
                                idx_ref_list.extend(["Abstract-"+article_data['metadata']['article_ids']['pmc']])
        
                            except:
                                print("File with PMC-ID %s can't append a text field!"%article_data['metadata']['article_ids']['pmc'])
                    file_in.close()

            elif entry.is_dir():
                # Recursivly call this function until dir_path is a file...
                ret_t_l, ref_l = add_embeddings_and_text_to_nparray_from_dir(entry.path, article_filter)
                ret_text_list.extend(ret_t_l)
                idx_ref_list.extend(ref_l)

        # Added all embeddings to ret_list and filenames to ref_list
        return np.array(ret_text_list), np.array(idx_ref_list, dtype=object)
