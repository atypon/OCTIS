import os

import numpy as np

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


def create_params(model_name, configs):
    parameters = {}
    if model_name == 'ProdLDA':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir
        }
    elif model_name == 'Scholar':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir
        }
    elif model_name == 'CTM':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir
        }
    elif model_name == 'SuperCTM':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir
        }
    elif model_name == 'NVDM':
        parameters = {
            'num_topics': configs.num_topics,
            'num_epochs': configs.epochs,
            'save_dir': configs.save_dir
        }

    return parameters
