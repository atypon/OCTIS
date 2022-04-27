import torch
from torch.utils.data import Dataset
import scipy.sparse


class CTMDataset(Dataset):
    """Class to load BOW dataset."""

    def __init__(self, X, X_bert, idx2token, ids=None, labels=None):
        """
        Args
            X : array-like, shape=(n_samples, n_features)
                Document word matrix.
        """
        if X.shape[0] != len(X_bert):
            raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                            "You might want to check if the BoW preparation method has removed some documents. ")
        if labels is not None:
            if labels.shape[0] != X.shape[0]:
                raise Exception(f"There is something wrong in the length of the labels (size: {labels.shape[0]}) "
                                f"and the bow (len: {X.shape[0]}). These two numbers should match.")

        self.X = X
        self.X_bert = X_bert
        self.idx2token = idx2token
        self.labels = labels
        self.ids = ids

    def __len__(self):
        """Return length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        if type(self.X[i]) == scipy.sparse.csr.csr_matrix:
            X = torch.FloatTensor(self.X[i].todense())
            X_bert = torch.FloatTensor(self.X_bert[i])
        else:
            X = torch.FloatTensor(self.X[i])
            X_bert = torch.FloatTensor(self.X_bert[i])

        return_dict = {'X': X, 'X_bert': X_bert}
        if self.labels is not None:
            labels = self.labels[i]
            if type(labels) == scipy.sparse.csr.csr_matrix:
                return_dict["labels"] = torch.FloatTensor(labels.todense())
            else:
                return_dict["labels"] = torch.FloatTensor(labels)
        if self.ids is not None:
            return_dict["ids"] = self.ids

        return return_dict
