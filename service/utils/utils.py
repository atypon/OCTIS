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
