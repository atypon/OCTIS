import os
import json
import codecs
import pickle

import gensim
import numpy as np
from scipy import sparse


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)


def read_json(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    return data


def read_jsonlist(input_filename):
    data = []
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            data.append(json.loads(line))
    return data


def write_jsonlist(list_of_json_objects, output_filename, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        for obj in list_of_json_objects:
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + '\n')


def pickle_data(data, output_filename):
    with open(output_filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


def unpickle_data(input_filename):
    with open(input_filename, 'rb') as infile:
        data = pickle.load(infile)
    return data


def read_text(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    return lines


def write_list_to_text(lines, output_filename, add_newlines=True, add_final_newline=False):
    if add_newlines:
        lines = '\n'.join(lines)
        if add_final_newline:
            lines += '\n'
    else:
        lines = ''.join(lines)
        if add_final_newline:
            lines[-1] += '\n'

    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.writelines(lines)


def save_sparse(sparse_matrix, output_filename):
    assert sparse.issparse(sparse_matrix)
    if sparse.isspmatrix_coo(sparse_matrix):
        coo = sparse_matrix
    else:
        coo = sparse_matrix.tocoo()
    row = coo.row
    col = coo.col
    data = coo.data
    shape = coo.shape
    np.savez(output_filename, row=row, col=col, data=data, shape=shape)


def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
    return coo_matrix.tocsc()


def load_word_vectors(word2vec_file, emb_dim, rng, vocab):
    if word2vec_file is not None:
        vocab_size = len(vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        embeddings = np.array(rng.rand(emb_dim, vocab_size) * 0.25 - 0.5, dtype=np.float32)
        count = 0
        print("Loading word vectors")
        # load the word2vec vectors
        pretrained = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)

        # replace the randomly initialized vectors with the word2vec ones for any that are available
        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[:, index] = pretrained[word]

        print("Found embeddings for %d words" % count)
        update_embeddings = False
    else:
        embeddings = None
        update_embeddings = True

    return embeddings, update_embeddings


def save_model_output(model_output, run, save_dir):
    if save_dir is not None:
        save_dir = os.path.join(save_dir, str(run))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
        for key, value in model_output.items():
            save_path = os.path.join(save_dir, key + '.txt')
            if key == 'topics':
                np.savetxt(save_path, np.array(value), fmt='%s')
            else:
                np.savetxt(save_path, value)
            print(save_path)
