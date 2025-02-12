import os
import json
import sys
import numpy as np
import torch
from tqdm.autonotebook import tqdm


def calculate_embedding_and_save(filepath_in, filepath_out, seq_len, seq_overlap, model, tokenizer=None, chunk_fnc='fixed', embedding_types=['title', 'abstract', 'article'], batchsize=100):
    '''
    POTENTIAL ERROR WITH BATCHES

    Read jsonl file with the relative filepath defined by 'filepath_in'.
    Call the function defined by 'model_fnc' with 'batchsize' count of text extracted from the jsonl to get embeddings.
    Subsequently save the original data in the directory 'filepath_out' with the added embeddings as a jsonl file.

    Parameters:
    - filepath_in (String): The relative Filepath to the file to be encoded.
    - filepath_out (String): The relative Filepath the the destination.
    - seq_len (int): Maximum amount of words accepted by the model. Additional words will be trunkated.
    - seq_overlap (int): Amount of words to overlap when creating chunks. Can be set to 0.
    - model (pytorch/transformer): A model to be used for the calculation of the embeddings.
    - tokenizer (for huggingface models): A Tokenizer to preprocess input for huggingface transformer models.
    - chunk_fnc (String, optional): Standard value='fixed'. A String representing the chunking function to be used for the model.
        ('fixed', 'sentence')
    - embedding_types (array, optional): 'title', 'abstract', 'article' can be added to list to calculate embeddings
    - batchsize (int, optional): Standard value='100'. Batchsize for writing to the file and article count for trunkated embedding.

    Returns:
    - None
    '''

    batched_articles = []
    title_embeddings, abstract_embeddings, text_embeddings = [], [], []

    # Check if output file exists and clear if it does
    if os.path.exists(filepath_out):
        try:
            open(filepath_out, 'w').close()
        except OSError:
            print ("Could not write file: ", filepath_out)
            sys.exit()
    
    # Check if cuda available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Read in Jsonl file
    try:
        file_in = open(filepath_in, 'r', encoding='utf-8')
    except OSError:
        print ("Could not open/read file: ", filepath_in)
        sys.exit()

    with file_in:
        for line in tqdm(file_in, desc="Reading lines"):
            article_data = json.loads(line)
            if 'title' in embedding_types:
                article_title = article_data['metadata']['article_title'] or ""
            else:
                article_title = " "
            if 'abstract' in embedding_types:
                article_abstract = article_data['text_content']['abstract'] or ""
            else:
                article_abstract = " "
            if 'article' in embedding_types:
                article_text = article_data['text_content']['article'] or ""
            else:
                article_text = " "

            # Calculate Embeddings for complete articles
            if article_title != "" and article_abstract != "" and article_abstract != "xx" and article_text != "":
                # Non corrupted data - Add to list
                batched_articles.append(article_data)

                # Abstracts and Titles are Calculated in batches and automaticly trunkated if necessary
                title_embeddings.append(article_title)
                abstract_embeddings.append(article_abstract)

                # The article text is divided into chunks and then embedded + combined
                if 'article' in embedding_types:
                    if chunk_fnc == 'fixed':
                        text_chunks = fixed_size_chunking(article_text, max_words=seq_len, overlap=seq_overlap)
                    elif chunk_fnc == 'sentence':
                        text_chunks = sentence_chunking(article_text,  max_words=seq_len)
                    else:
                        raise(Exception("An unknown chunking function was specified!"))  
                
                    # Calculate embeddings for all chunks and combine
                    text_embeddings.append(sentence_transformer_encode(input=text_chunks, model=model))
                else:
                    text_embeddings.append(np.array([]))

                # Make sure can be indexed. Remove later
                assert len(title_embeddings) == len(abstract_embeddings) \
                        and len(title_embeddings) == len(batched_articles) 

                # Batchsize for the model reached
                # Calculate Embeddings for title and abstract + write to file
                if len(title_embeddings) >= batchsize:
                    if 'title' in embedding_types:
                        title_embeddings = sentence_transformer_encode(input=title_embeddings, model=model)
                    else:
                        title_embeddings = [np.array([])] * batchsize
                    if 'abstract' in embedding_types:
                        abstract_embeddings = sentence_transformer_encode(input=abstract_embeddings, model=model)
                    else:
                        abstract_embeddings = [np.array([])] * batchsize
                    
                    batched_articles = add_embeddings_to_dict(batched_articles, title_embeddings, abstract_embeddings, text_embeddings)

                    # Write batched_articles to file
                    try:
                        file_out = open(filepath_out, 'a', encoding='utf-8')
                    except OSError:
                        print ("Could not open/write file: ", filepath_out)
                        sys.exit()
                    with file_out:
                        for item in batched_articles:
                            #print("Successfully embedded: " + item['filename'])
                            file_out.write(json.dumps(item) + '\n')
                    file_out.close()

                    # Clear variables
                    batched_articles = []
                    title_embeddings = []
                    abstract_embeddings = []

            else:
                try:
                    file_err = open(os.path.dirname(filepath_out) + '/FailLog.txt', 'a', encoding='utf-8')
                except OSError:
                    print ("Could not open/write file: ", (os.path.dirname(filepath_out) + '/FailLog.txt'))
                    sys.exit()
                with file_err:
                    #print("Failed to embedd: " + article_data['filename'])
                    file_err.write(f"Articel with filename '{article_data['filename']}' could not be embedded. Some field is missing (Title, Abstract or Article-Text)\n")
                file_err.close()
            
    # Can write file at the end and save in 'embedded_articles' until then...
    # Calculation for the rest of the data
    if 'title' in embedding_types:
        title_embeddings = sentence_transformer_encode(input=title_embeddings, model=model)
    else:
        title_embeddings = [np.array([])] * batchsize
    if 'abstract' in embedding_types:
        abstract_embeddings = sentence_transformer_encode(input=abstract_embeddings, model=model)
    else:
        abstract_embeddings = [np.array([])] * batchsize
    
    batched_articles = add_embeddings_to_dict(batched_articles, title_embeddings, abstract_embeddings, text_embeddings)

    # Write batched articles to file   
    try:
        file_out = open(filepath_out, 'a', encoding='utf-8')
    except OSError:
        print ("Could not open/write file: ", filepath_out)
        sys.exit()
    with file_out:
        for item in batched_articles:
            #print("Successfully embedded: " + item['filename'])
            file_out.write(json.dumps(item) + '\n')
    file_out.close()
    file_in.close()
    print("\nFile: ",filepath_in, " has succesfully beeen embedded.\n------------------------------------------\n")


def sentence_transformer_encode(input, model):
    '''
    Calculating the embeddings for the given Strings when using a sentencetransformer model.
    Combining all Embeddings into a onedimensional embedding using specified method (avg or max).

    Parameters:
    - input (List<String>): A List/Array of the Strings to be fed to the model. It will trunkate any excess text.
    - model (SentenceTransformer): A model to be used for the calculation.

    Returns:
    - Embeddings (List): List of embeddings...
    '''
    output = model.encode(input)
    return output


def add_embeddings_to_dict(batched_articles, title_embeddings, abstract_embeddings, text_embeddings):
    '''
    Add the given embeddings in a new dict column called embeddings.

    Parameters:
    - batched_articles (List<dict>): List of dictionaries containting raw article data.
    - <>_embeddings (List<Embeddings>): List of Embeddings for each part (title, abstract, article-text)

    Returns:
    - batched_articles (List<dict>): Return batched_articles with added column/data.
    '''

    for i in range(len(batched_articles)):
        batched_articles[i]['embeddings'] = {}
        batched_articles[i]['embeddings']['title'] = title_embeddings[i].tolist()
        batched_articles[i]['embeddings']['abstract'] = abstract_embeddings[i].tolist()
        batched_articles[i]['embeddings']['article'] = text_embeddings[i].tolist()
    return batched_articles


def loop_through_directory(dir_path, dir_path_out, seq_len, seq_overlap, model, tokenizer=None, chunk_fnc='fixed', embedding_types=['title', 'abstract', 'article'], batchsize=100):
    '''
    Loop through a directory given by 'dir_path' and call function 'calculate_embedding_and_save' on all files in it and in any subdirectory.

    Parameters:
    - dir_path (String): Path to the directory one wants to loop through.
    - dir_path_out (String): Path to the directory of where to save results after function call.
    - Refer to function description of calculate_embedding_and_save() for detailed meanning of parameters.
    -...

    Returns:
    - None
    '''
    # Check if dir_path_out exists
    if not os.path.exists(dir_path_out):
        os.makedirs(dir_path_out)

    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_file():
                filepath_out = dir_path_out + os.path.splitext(os.path.basename(entry.path))[0] + '-Embedded.jsonl'
                calculate_embedding_and_save(entry.path, filepath_out, seq_len, seq_overlap, model, tokenizer, chunk_fnc, embedding_types, batchsize)
            elif entry.is_dir():
                loop_through_directory(entry.path, dir_path_out, seq_len, seq_overlap, model, tokenizer, chunk_fnc, embedding_types, batchsize)


def fixed_size_chunking(input_string, max_words, overlap):
    '''
    Chunking based on a given maximum word length and overlap.
    It uses the tokenization_fnc to calculate and subsequently return tokenized chunks.
    
    Parameters:
    - ...

    Returns:
    - List<String>: The input_string parted in a List of chunks of defined size and overlap. 
    '''
    input_words = input_string.split(' ')
    chunks = [' '.join(input_words[i:i+max_words]) for i in range(0, len(input_words), max_words-overlap)]
    return chunks


def sentence_chunking(input_string, max_words=None):
    '''
    Chunking the text, whenever a '.' is found. 
    If max_words and overlap are provided and a sentence is longer than max_words,
     it will recusively split the sentence until chunks smaller than max_words are created

    Parameters:
    - input_string (String): Input text to be chunked.
    - max_words (int, optional): Used when want to avoid trunkation.

    Returns:
    - List<String>: The input_string parted in a List of chunks containing sentences.
    '''
    sentences = input_string.split('.')
    if max_words != None:
        c = 0
        while c < len(sentences):
            s = sentences[c].split(' ')
            if len(s) > max_words:
                s1 = ' '.join(s[0:len(s)//2])
                s2 = ' '.join(s[(len(s)//2)+1:-1])
                del sentences[c]
                sentences.append(s1)
                sentences.append(s2)
            else:
                c+=1
    return sentences