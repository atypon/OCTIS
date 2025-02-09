from sklearn.preprocessing import MultiLabelBinarizer

from octis.models.contextualized_topic_models.datasets.dataset import CTMDataset
from sklearn.feature_extraction.text import CountVectorizer
import multiprocessing as mp
from octis.models.model import AbstractModel
from octis.models.contextualized_topic_models.datasets import dataset
from octis.models.contextualized_topic_models.models import ctm
from octis.models.contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list

import os
import time
import pickle as pkl
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CTM(AbstractModel):

    def __init__(self, num_topics=10, ctm_type='CTM', model_type='prodLDA', activation='softplus',
                 dropout=0.2, learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99, solver='adam',
                 num_epochs=100, reduce_on_plateau=False, prior_mean=0.0, prior_variance=None, num_layers=2,
                 num_neurons=100, use_partitions=True, num_samples=10, inference_type="combined", bert_path="",
                 test_bert_path="", bert_model="all-mpnet-base-v2", loss_weights=None, save_dir=None):
        """
        initialization of CTM

        :param num_topics : int, number of topic components, (default 10)
        :param ctm_type : string, 'CTM' or 'SuperCTM' (default 'CTM')
        :param model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
        :param activation : string, 'softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu', 'rrelu', 'elu',
        'selu' (default 'softplus')
        :param num_layers : int, number of layers (default 2)
        :param solver : string, optimizer 'adam' or 'sgd' (default 'adam')
        :param dropout : float, dropout to use (default 0.2)
        :param learn_priors : bool, make priors a learnable parameter (default True)
        :param batch_size : int, size of batch to use for training (default 64)
        :param lr : float, learning rate to use for training (default 2e-3)
        :param momentum : float, momentum to use for training (default 0.99)
        :param num_epochs : int, number of epochs to train for, (default 100)
        :param num_samples: int, number of times theta needs to be sampled (default: 10)
        :param use_partitions: bool, if true the model will be trained on the training set and evaluated on the test
        set (default: true)
        :param reduce_on_plateau : bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
        :param inference_type: the type of the CTM model. It can be "zeroshot" or "combined" (default zeroshot)
        :param bert_path: path to store the document contextualized representations
        :param test_bert_path: path to store the test document contextualized representations
        :param bert_model: name of the contextualized model (default: bert-base-nli-mean-tokens).
        :param prior_mean: float
        :param prior_variance: float
        :param num_neurons:int
        :param loss_weights:
        :param save_dir: string
        see https://www.sbert.net/docs/pretrained_models.html
        """

        super().__init__()

        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['model_type'] = model_type
        self.hyperparameters['activation'] = activation
        self.hyperparameters['dropout'] = dropout
        self.hyperparameters['inference_type'] = inference_type
        self.hyperparameters['learn_priors'] = learn_priors
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['lr'] = lr
        self.hyperparameters['num_samples'] = num_samples
        self.hyperparameters['momentum'] = momentum
        self.hyperparameters['solver'] = solver
        self.hyperparameters['num_epochs'] = num_epochs
        self.hyperparameters['reduce_on_plateau'] = reduce_on_plateau
        self.hyperparameters["prior_mean"] = prior_mean
        self.hyperparameters["prior_variance"] = prior_variance
        self.hyperparameters["num_neurons"] = num_neurons
        self.hyperparameters["bert_path"] = bert_path
        self.hyperparameters["test_bert_path"] = test_bert_path
        self.hyperparameters["num_layers"] = num_layers
        self.hyperparameters["bert_model"] = bert_model
        self.hyperparameters["save_dir"] = save_dir

        self.use_partitions = use_partitions
        self.ctm_type = ctm_type

        hidden_sizes = tuple([num_neurons for _ in range(num_layers)])
        self.hyperparameters['hidden_sizes'] = tuple(hidden_sizes)
        self.hyperparameters['loss_weights'] = loss_weights

        self.model = None
        self.vocab = None

    def train_model(self, dataset, hyperparameters=None, top_words=10):
        """
        trains CTM model

        :param dataset: octis Dataset for training the model
        :param hyperparameters: dict, with optionally) the following information:
        :param top_words: number of top-n words of the topics (default 10)

        """
        if hyperparameters is None:
            hyperparameters = {}
        self.set_params(hyperparameters)

        x_train, x_test, x_bow, x_test_bow, x_valid, input_size = self.preprocess(
            dataset=dataset,
            bert_model=self.hyperparameters["bert_model"],
            bert_path=self.hyperparameters["bert_path"],
            test_bert_path=self.hyperparameters["test_bert_path"],
            partition=self.use_partitions,
            use_validation=True)

        self.model = ctm.CTM(input_size=input_size,
                             bert_input_size=x_train.X_bert.shape[1],
                             label_size=0 if self.ctm_type == 'CTM' else x_train.labels.shape[1],
                             inference_type=self.hyperparameters['inference_type'],
                             model_type='prodLDA',
                             n_components=self.hyperparameters['num_topics'],
                             dropout=self.hyperparameters['dropout'],
                             activation=self.hyperparameters['activation'],
                             lr=self.hyperparameters['lr'],
                             hidden_sizes=self.hyperparameters['hidden_sizes'],
                             solver=self.hyperparameters['solver'],
                             momentum=self.hyperparameters['momentum'],
                             num_epochs=self.hyperparameters['num_epochs'],
                             learn_priors=self.hyperparameters['learn_priors'],
                             batch_size=self.hyperparameters['batch_size'],
                             reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
                             loss_weights=self.hyperparameters['loss_weights'])

        if self.use_partitions:
            self.model.fit(x_train, x_valid, verbose=False, save_dir=self.hyperparameters["save_dir"])
            result = self.inference(x_test, x_bow, x_test_bow)
            return result

        else:
            self.model.fit(x_train, None, verbose=False, save_dir=self.hyperparameters["save_dir"])
            result = self.model.get_info()
            return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys() and k != 'hidden_sizes':
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

        self.hyperparameters['hidden_sizes'] = tuple(
            [self.hyperparameters["num_neurons"] for _ in range(self.hyperparameters["num_layers"])])

    def inference(self, x_test, x_bow, x_test_bow):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(x_test, x_bow, x_test_bow)
        return results

    def predict(self):
        info = {}
        topic_word = self.model.get_topic_lists()
        topic_word_dist = self.model.get_topic_word_matrix()

        info['topics'] = topic_word
        info['topic-document-matrix'] = np.asarray(self.model.get_thetas(self.model.train_data)).T
        info['topic-word-matrix'] = topic_word_dist

        return info

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    def preprocess(self, dataset, bert_model, partition=True, use_validation=True, bert_path=None, test_bert_path=None):
        doc_ids, unprocessed_docs, preprocessed_docs, labels, metadata = self.filter_empty_labels(dataset)
        training, validation, testing, train_bow, test_bow = None, None, None, None, None

        # unprocessed_docs = [' '.join(x) for x in unprocessed_docs]
        preprocessed_docs = [' '.join(x) for x in preprocessed_docs]

        vocab = dataset.get_vocabulary()
        vocab2id = {w: i for i, w in enumerate(vocab)}
        vectorizer = CountVectorizer(vocabulary=vocab2id)
        vectorizer.fit(preprocessed_docs)
        idx2token = {v: k for (k, v) in vectorizer.vocabulary_.items()}
        input_size = len(idx2token.keys())

        if labels:
            label_encoder = MultiLabelBinarizer()
            encoded_labels = label_encoder.fit_transform(np.array(labels))
        else:
            encoded_labels = None

        if bert_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        # bow_embeddings = vectorizer.transform(preprocessed_docs)
        contextualized_embeddings = self.load_bert_data(bert_path, unprocessed_docs, bert_model, ids=doc_ids)

        train_data, test_data, valid_data = None, None, None

        if partition:
            train_data, valid_data, test_data = dataset.get_partitioned_corpus(use_validation=use_validation)

            if train_data is not None:
                last_training_doc = metadata["last-training-doc"]
                train_data = [' '.join(i) for i in train_data]
                # train_bow_embeddings = bow_embeddings[:last_training_doc]
                train_bow_embeddings = vectorizer.transform(train_data)
                train_contextualized_embeddings = contextualized_embeddings[:last_training_doc]
                train_labels = encoded_labels[:last_training_doc] if labels else None
                train_ids = doc_ids[:last_training_doc] if doc_ids else None
                training = CTMDataset(
                    train_bow_embeddings, train_contextualized_embeddings, idx2token, train_ids, train_labels)

            if valid_data is not None:
                last_validation_doc = metadata["last-validation-doc"]
                valid_data = [' '.join(i) for i in valid_data]
                # v_bow_embeddings = bow_embeddings[last_training_doc:last_validation_doc]
                v_bow_embeddings = vectorizer.transform(valid_data)
                v_contextualized_embeddings = contextualized_embeddings[last_training_doc:last_validation_doc]
                v_encoded_labels = encoded_labels[last_training_doc:last_validation_doc] if labels else None
                v_ids = doc_ids[last_training_doc:last_validation_doc] if doc_ids else None
                validation = CTMDataset(
                    v_bow_embeddings, v_contextualized_embeddings, idx2token, v_ids, v_encoded_labels)

            if test_data is not None:

                if "last-validation-doc" in metadata:
                    last_validation_doc = metadata["last-validation-doc"]
                    test_index = last_validation_doc
                else:
                    test_index = last_training_doc

                test_data = [' '.join(i) for i in test_data]
                te_bow_embeddings = vectorizer.transform(test_data)
                te_contextualized_embeddings = contextualized_embeddings[test_index:]

                test_list = [t.split(' ') for t in test_data]
                test_ids = doc_ids[test_index:]
                train_bow = [' '.join([t[x] for x in range(1, len(t), 2)]) for t in test_list]
                test_bow = [' '.join([t[x] for x in range(1, len(t), 2)]) for t in test_list]

                train_bow_embeddings = vectorizer.transform(train_bow)
                train_bow_contextualized_embeddings = self.load_bert_data(test_bert_path, train_bow, bert_model,
                                                                          ids=test_ids)
                test_bow_embeddings = vectorizer.transform(test_bow)
                test_bow_contextualized_embeddings = self.load_bert_data(test_bert_path, test_bow, bert_model,
                                                                         ids=test_ids)
                if labels:
                    te_encoded_labels = encoded_labels[test_index:]
                else:
                    te_encoded_labels = None

                t_ids = doc_ids[test_index:] if doc_ids else None
                testing = CTMDataset(
                    te_bow_embeddings, te_contextualized_embeddings, idx2token, t_ids, te_encoded_labels)
                train_bow = CTMDataset(
                    train_bow_embeddings, train_bow_contextualized_embeddings, idx2token, t_ids, te_encoded_labels)
                test_bow = CTMDataset(
                    test_bow_embeddings, test_bow_contextualized_embeddings, idx2token, t_ids, te_encoded_labels)

        else:
            bow_embeddings = vectorizer.transform(preprocessed_docs)
            training = CTMDataset(bow_embeddings, contextualized_embeddings, idx2token, doc_ids, encoded_labels)

        return training, testing, train_bow, test_bow, validation, input_size

    def filter_empty_labels(self, dataset):
        doc_ids = dataset.get_document_indexes()
        unprocessed_docs = dataset.get_raw_corpus()
        preprocessed_docs = dataset.get_corpus()
        metadata = dataset.get_metadata()
        if self.ctm_type == 'CTM':
            labels = None
        else:
            labels = dataset.get_labels()

        if labels:
            doc_ids = [p for i, p in enumerate(doc_ids) if labels[i]]
            unprocessed_docs = [p for i, p in enumerate(unprocessed_docs) if labels[i]]
            preprocessed_docs = [p for i, p in enumerate(preprocessed_docs) if labels[i]]
            labels = [p for i, p in enumerate(labels) if labels[i]]

        return doc_ids, unprocessed_docs, preprocessed_docs, labels, metadata

    @staticmethod
    def load_bert_data(bert_path, texts, bert_model, ids=None):
        if bert_path is not None:
            if os.path.exists(bert_path):
                bert_output = pkl.load(open(bert_path, 'rb'))
                temp_output = {}
                for output in bert_output:
                    temp_output[output['id']] = output['embedding']
                bert_output = np.array([temp_output[id] for id in ids])
            else:
                bert_output = bert_embeddings_from_list(texts, bert_model)
                save_output = bert_output
                if ids is not None:
                    save_output = [{'id': x, 'embedding': y} for x, y in zip(ids, bert_output)]
                if bert_path:
                    bert_path = bert_path if bert_path.endswith('.pkl') else bert_path + '_' + time.strftime(
                        '%Y%m%d%H%M') + '.pkl'
                    pkl.dump(save_output, open(bert_path, 'wb'))
                else:
                    bert_path = 'bert_model_' + time.strftime('%Y%m%d%H%M') + '.pkl'
                    pkl.dump(save_output, open(bert_path, 'wb'))
        else:
            bert_output = bert_embeddings_from_list(texts, bert_model)
        return bert_output
