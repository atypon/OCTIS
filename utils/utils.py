import os

import numpy as np

import utils.file_handling as fh


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


def save_document_embed(model_output, ids, run, save_dir):
    if save_dir is not None:
        save_dir = os.path.join(save_dir, str(run))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
        if 'topic-document-matrix' in model_output:
            save_path = os.path.join(save_dir, 'topic-document-matrix_embed' + '.txt')
            document_embed = model_output['topic-document-matrix'].T
            results = []
            for i, j in zip(ids, document_embed):
                results.append({"paper_id": i, "title": "", "embedding": j.tolist()})
            fh.write_jsonlist(results, save_path)
            print(save_path)


def create_params(model_name, configs):
    parameters = {}
    if model_name == 'ProdLDA':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir,
            'use_partitions': configs.use_partitions
        }
    elif model_name == 'Scholar':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir,
            'use_partitions': configs.use_partitions
        }
    elif model_name == 'CTM':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir,
            'bert_path': configs.training_embeddings,
            'test_bert_path': configs.testing_embeddings,
            'ctm_type': 'CTM',
            'use_partitions': configs.use_partitions
        }
    elif model_name == 'SuperCTM':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir,
            'bert_path': configs.training_embeddings,
            'test_bert_path': configs.testing_embeddings,
            'ctm_type': 'SuperCTM',
            'use_partitions': configs.use_partitions
        }
    elif model_name == 'NVDM':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir
        }

    return parameters
