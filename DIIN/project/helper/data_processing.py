
import numpy as np
import re
import random
import json
import collections
import numpy as np
import helper.p.p.parameters as params
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn 
import os
import pickle
import multiprocessing
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger

FIXED_PARAMETERS, config = params.load_parameters()

LABEL_MAP = {
    "entailment": 1,
    "nonentailment": 0
    # "neutral": 1,
    # "contradiction": 2,
    # "hidden": -1
}

PADDING = "<PAD>"
POS_Tagging = [PADDING, 'WP$', 'RBS', 'SYM', 'WRB', 'IN', 'VB', 'POS', 'TO', ':', '-RRB-', '$', 'MD', 'JJ', '#', 'CD', '``', 'JJR', 'NNP', "''", 'LS', 'VBP', 'VBD', 'FW', 'RBR', 'JJS', 'DT', 'VBG', 'RP', 'NNS', 'RB', 'PDT', 'PRP$', '.', 'XX', 'NNPS', 'UH', 'EX', 'NN', 'WDT', 'VBN', 'VBZ', 'CC', ',', '-LRB-', 'PRP', 'WP']
POS_dict = {pos:i for i, pos in enumerate(POS_Tagging)}

base_path = os.getcwd()
nltk_data_path = base_path + "/../TF/nltk_data"
nltk.data.path.append(nltk_data_path)
stemmer = nltk.SnowballStemmer('english')

tt = nltk.tokenize.treebank.TreebankWordTokenizer()


def construct_one_hot_feature_tensor(sequences, left_padding_and_cropping_pairs, dim, column_size=None, dtype=np.int32):
    """
    sequences: [[(idx, val)... ()]...[]]
    left_padding_and_cropping_pairs: [[(0,0)...] ... []]
    """
    tensor_list = []
    for sequence, pad_crop_pair in zip(sequences, left_padding_and_cropping_pairs):
        left_padding, left_cropping = pad_crop_pair
        if dim == 1:
            vec = np.zeros((config.seq_length))
            for num in sequence:
                if num + left_padding - left_cropping < config.seq_length and num + left_padding - left_cropping >= 0:
                    vec[num + left_padding - left_cropping] = 1
            tensor_list.append(vec)
        elif dim == 2:
            assert column_size
            mtrx = np.zeros((config.seq_length, column_size))
            for row, col in sequence:
                if row + left_padding - left_cropping < config.seq_length and row + left_padding - left_cropping >= 0 and col < column_size:
                    mtrx[row + left_padding - left_cropping, col] = 1
            tensor_list.append(mtrx)

        else:
            raise NotImplementedError

    return np.array(tensor_list, dtype=dtype)


def load_nli_data(path, snli=False, shuffle = True):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path) as f:
        for line in tqdm(f):
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    return data

def is_exact_match(token1, token2):
    token1 = token1.lower()
    token2 = token2.lower()

    token1_stem = stemmer.stem(token1)

    if token1 == token2:
        return True
    
    for synsets in wn.synsets(token2):
        for lemma in synsets.lemma_names():
            if token1_stem == stemmer.stem(lemma):
                return True
    
    if token1 == "n't" and token2 == "not":
        return True
    elif token1 == "not" and token2 == "n't":
        return True
    elif token1_stem == stemmer.stem(token2):
        return True
    return False

def is_antonyms(token1, token2):
    token1 = token1.lower()
    token2 = token2.lower()
    token1_stem = stemmer.stem(token1)
    antonym_lists_for_token2 = []
    for synsets in wn.synsets(token2):
        for lemma_synsets in [wn.synsets(l) for l in synsets.lemma_names()]:
            for lemma_syn in lemma_synsets:
                for lemma in lemma_syn.lemmas():
                    for antonym in lemma.antonyms():
                        antonym_lists_for_token2.append(antonym.name())
                        # if token1_stem == stemmer.stem(antonym.name()):
                        #     return True 
    antonym_lists_for_token2 = list(set(antonym_lists_for_token2))
    for atnm in antonym_lists_for_token2:
        if token1_stem == stemmer.stem(atnm):
            return True
    return False   




def load_mnli_shared_content():
    shared_file_exist = False

    shared_path = config.datapath + "/data.jsonl"
    print(shared_path)

    
    if os.path.isfile(shared_path):
        shared_file_exist = True

    with open(shared_path) as f:
        shared_content = {}
        # load_shared_content(f, shared_content)
        shared_content = json.load(f)
    return shared_content

def sentences_to_padded_index_sequences(datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    # Extract vocabulary
    shared_content = {}
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    
    shared_file_exist = False
     
    word_counter = collections.Counter()
    char_counter = collections.Counter()
    mgr = multiprocessing.Manager()
    shared_content = mgr.dict()
    process_num = config.num_process_prepro
    process_num = 10
    for i, dataset in enumerate(datasets):
     
        for example in tqdm(dataset):
            s1_tokenize = tokenize(example['sentence1_binary_parse'])
            s2_tokenize = tokenize(example['sentence2_binary_parse'])
 
            word_counter.update(s1_tokenize)
            word_counter.update(s2_tokenize)

            for i, word in enumerate(s1_tokenize):
                char_counter.update([c for c in word])
            for word in s2_tokenize:
                char_counter.update([c for c in word])

#
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    if config.embedding_replacing_rare_word_with_UNK: 
        vocabulary = [PADDING, "<UNK>"] + vocabulary
    else:
        vocabulary = [PADDING] + vocabulary
    # print(char_counter)
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    indices_to_words = {v: k for k, v in word_indices.items()}
    char_vocab = set([char for char in char_counter])
    char_vocab = list(char_vocab)
    char_vocab = [PADDING] + char_vocab
    char_indices = dict(zip(char_vocab, range(len(char_vocab))))
    indices_to_char = {v: k for k, v in char_indices.items()}
    

    for i, dataset in enumerate(datasets):
        for example in tqdm(dataset):
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
                example[sentence + '_inverse_term_frequency'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.float32)

                token_sequence = tokenize(example[sentence])
                padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)
                      
                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                        itf = 0
                    else:
                        if config.embedding_replacing_rare_word_with_UNK:
                            index = word_indices[token_sequence[i]] if word_counter[token_sequence[i]] >= config.UNK_threshold else word_indices["<UNK>"]
                        else:
                            index = word_indices[token_sequence[i]]
                        itf = 1 / (word_counter[token_sequence[i]] + 1)
                    example[sentence + '_index_sequence'][i] = index
                    
                    example[sentence + '_inverse_term_frequency'][i] = itf
                
                example[sentence + '_char_index'] = np.zeros((FIXED_PARAMETERS["seq_length"], config.char_in_word_size), dtype=np.int32)
                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        continue
                    else:
                        chars = [c for c in token_sequence[i]]
                        for j in range(config.char_in_word_size):
                            if j >= (len(chars)):
                                break
                            else:
                                index = char_indices[chars[j]]
                            example[sentence + '_char_index'][i,j] = index 
    

    return indices_to_words, word_indices, char_indices, indices_to_char

def parsing_parse(parse):
    base_parse = [s.rstrip(" ").rstrip(")") for s in parse.split("(") if ")" in s]
    pos = [pair.split(" ")[0] for pair in base_parse]
    return pos

def parse_to_pos_vector(parse, left_padding_and_cropping_pair = (0,0)): # ONE HOT
    pos = parsing_parse(parse)
    pos_vector = [POS_dict.get(tag,0) for tag in pos]
    left_padding, left_cropping = left_padding_and_cropping_pair
    vector = np.zeros((FIXED_PARAMETERS["seq_length"],len(POS_Tagging)))
    assert left_padding == 0 or left_cropping == 0

    for i in range(FIXED_PARAMETERS["seq_length"]):
        if i < len(pos_vector):
            vector[i + left_padding, pos_vector[i + left_cropping]] = 1
        else:
            break
    return vector

def generate_pos_feature_tensor(parses, left_padding_and_cropping_pairs):
    pos_vectors = []
    for parse in parses:
        pos = parsing_parse(parse)
        pos_vector = [(idx, POS_dict.get(tag, 0)) for idx, tag in enumerate(pos)]
        pos_vectors.append(pos_vector)

    return construct_one_hot_feature_tensor(pos_vectors, left_padding_and_cropping_pairs, 2, column_size=len(POS_Tagging))


def fill_feature_vector_with_cropping_or_padding(sequences, left_padding_and_cropping_pairs, dim, column_size=None, dtype=np.int32):
    if dim == 1:
        list_of_vectors = []
        for sequence, pad_crop_pair in zip(sequences, left_padding_and_cropping_pairs):
            vec = np.zeros((config.seq_length))
            left_padding, left_cropping = pad_crop_pair
            for i in range(config.seq_length):
                if i + left_padding < config.seq_length and i - left_cropping < len(sequence):
                    vec[i + left_padding] = sequence[i + left_cropping]
                else:
                    break
            list_of_vectors.append(vec)
        return np.array(list_of_vectors, dtype=dtype)
    elif dim == 2:
        assert column_size
        tensor_list = []
        for sequence, pad_crop_pair in zip(sequences, left_padding_and_cropping_pairs):
            left_padding, left_cropping = pad_crop_pair
            mtrx = np.zeros((config.seq_length, column_size))
            for row_idx in range(config.seq_length):
                if row_idx + left_padding < config.seq_length and row_idx < len(sequence) + left_cropping:
                    for col_idx, content in enumerate(sequence[row_idx + left_cropping]):
                        mtrx[row_idx + left_padding, col_idx] = content
                else:
                    break
            tensor_list.append(mtrx)
        return np.array(tensor_list, dtype=dtype)
    else:
        raise NotImplementedError


