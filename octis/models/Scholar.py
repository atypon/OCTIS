import sys

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

from octis.models.scholar_model.scholar import Scholar
from sklearn.preprocessing import MultiLabelBinarizer
import utils.file_handling as fh


class scholar:
    def __init__(self, num_topics=10, batch_size=512, solver='Adam', activation='softplus',
                 lr=0.002, momentum=0.99, dropout=0.2, regularization=False,
                 num_epochs=100, init_mult=0.001, device=torch.device('cpu'),
                 classifier_layers=0, use_interactions=False, l1_topics=1.0, l1_topic_covars=1.0,
                 l1_interactions=1.0, l2_prior_covars=0.0, embedding_dim=300, use_partitions=True,
                 w2v=None, alpha=1.0, no_bg=False, seed=None, covars_predict=False, save_dir=None):

        self.hyperparameters = {}
        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['solver'] = solver
        self.hyperparameters['activation'] = activation
        self.hyperparameters['dropout'] = dropout
        self.hyperparameters['lr'] = lr
        self.hyperparameters['momentum'] = momentum

        self.hyperparameters['num_epochs'] = num_epochs
        self.hyperparameters['init_mult'] = init_mult
        self.hyperparameters['device'] = device
        self.hyperparameters['classifier_layers'] = classifier_layers
        self.hyperparameters['use_interactions'] = use_interactions
        self.hyperparameters['no_bg'] = no_bg
        self.hyperparameters['alpha'] = alpha
        self.hyperparameters['covars_predict'] = covars_predict
        self.hyperparameters['w2v'] = w2v
        self.hyperparameters["embedding_dim"] = embedding_dim
        self.use_partitions = use_partitions

        self.hyperparameters["l2_prior_covars"] = l2_prior_covars
        self.hyperparameters['regularization'] = regularization
        if regularization:
            self.hyperparameters['l1_topics'] = 1.0
            self.hyperparameters["l1_topic_covars"] = 1.0
            self.hyperparameters["l1_interactions"] = 1.0
        else:
            self.hyperparameters['l1_topics'] = l1_topics
            self.hyperparameters["l1_topic_covars"] = l1_topic_covars
            self.hyperparameters["l1_interactions"] = l1_interactions

        self.hyperparameters['seed'] = seed
        if self.hyperparameters['seed'] is not None:
            self.rng = np.random.RandomState(seed)
            self.seed = self.hyperparameters['seed']
        else:
            self.rng = np.random.RandomState(np.random.randint(0, 100000))
            self.seed = None

        self.hyperparameters['n_labels'] = 0
        self.hyperparameters['n_prior_covars'] = 0
        self.hyperparameters['n_topic_covars'] = 0

    def train_model(self, dataset, hyperparameters=None, top_words=10):
        """
        trains Scholar model

        :param dataset: octis Dataset for training the model
        :param hyperparameters: dict, with optionally) the following information:
        :param top_words: number of top-n words of the topics (default 10)
        """

        if hyperparameters is None:
            hyperparameters = {}
        self.set_params(hyperparameters)

        labels, covars = None, None
        mlb = MultiLabelBinarizer()
        if dataset.get_labels():
            labels = mlb.fit_transform(dataset.get_labels())
            self.hyperparameters['n_labels'] = len(mlb.classes_)

        if dataset.get_covariates():
            covars = mlb.fit_transform(dataset.get_covariates())
            self.hyperparameters['n_topic_covars'] = len(mlb.classes_)

        self.vocab = dataset.get_vocabulary()

        corpus = dataset.get_corpus()
        corpus = [' '.join(x) for x in corpus]
        vocab2id = {w: i for i, w in enumerate(self.vocab)}
        vec = CountVectorizer(vocabulary=vocab2id, token_pattern=r'(?u)\b\w+\b')
        vec.fit(corpus)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}

        if self.use_partitions:
            train_indices, test_indices, valid_indices = dataset.get_split_indices(use_validation=True)
            train_data, valid_data, test_data = self.split_data(np.array(corpus), train_indices, test_indices,
                                                                valid_indices)
            train_X = vec.transform(train_data) if train_data is not None else None
            valid_X = vec.transform(valid_data) if valid_data is not None else None
            test_X = vec.transform(test_data) if test_data is not None else None

            train_labels, valid_labels, test_labels = self.split_data(labels, train_indices, test_indices,
                                                                      valid_indices)
            train_prior_covars, valid_prior_covars, test_prior_covars = None, None, None
            train_topic_covars, valid_topic_covars, test_topic_covars = self.split_data(covars,
                                                                                        train_indices,
                                                                                        test_indices,
                                                                                        valid_indices)

            init_bg = self.get_init_bg(train_X.toarray().sum(axis=0))
            if self.hyperparameters['no_bg']:
                init_bg = np.zeros_like(init_bg)

            if test_data is not None:
                test_data = [d.split() for d in test_data]
                test_fh, test_sh = self.split_bow(test_data)
                test_fh_X, test_sh_X = vec.transform(test_fh), vec.transform(test_sh)
        else:
            train_X = vec.transform(corpus)
            train_labels = labels
            train_prior_covars = None
            train_topic_covars = covars

            init_bg = self.get_init_bg(train_X.toarray().sum(axis=0))
            if self.hyperparameters['no_bg']:
                init_bg = np.zeros_like(init_bg)

        embeddings, update_embeddings = fh.load_word_vectors(self.hyperparameters['w2v'],
                                                             self.hyperparameters['embedding_dim'],
                                                             self.seed,
                                                             self.vocab)

        network_architecture = self.make_network(self.hyperparameters,
                                                 len(idx2token.keys()),
                                                 self.hyperparameters['n_labels'],
                                                 self.hyperparameters['n_prior_covars'],
                                                 self.hyperparameters['n_topic_covars'])

        self.model = Scholar(network_architecture,
                             alpha=self.hyperparameters['alpha'],
                             learning_rate=self.hyperparameters['lr'],
                             init_embeddings=embeddings,
                             update_embeddings=update_embeddings,
                             init_bg=init_bg,
                             adam_beta1=self.hyperparameters['momentum'],
                             seed=self.seed,
                             classify_from_covars=self.hyperparameters['covars_predict'])

        print("Optimizing full model")
        if self.use_partitions:
            self.train(network_architecture,
                       train_X, train_labels, train_prior_covars, train_topic_covars,
                       training_epochs=self.hyperparameters['num_epochs'],
                       X_dev=valid_X, Y_dev=valid_labels, PC_dev=valid_prior_covars, TC_dev=valid_topic_covars)

            result = self.inference(train_X, train_labels, train_prior_covars, train_topic_covars,
                                    test_X, test_labels, test_prior_covars, test_topic_covars,
                                    test_fh_X, test_sh_X, batch_size=200, eta_bn_prop=1.0)
        else:
            self.train(network_architecture, train_X, train_labels, train_prior_covars, train_topic_covars,
                       training_epochs=self.hyperparameters['num_epochs'])
            result = self.inference(train_X, train_labels, train_prior_covars, train_topic_covars)
        return result

    def inference(self, train_X, train_labels, train_prior_covars, train_topic_covars,
                  test_X=None, test_labels=None, test_prior_covars=None, test_topic_covars=None,
                  test_bow_fh=None, test_bow_sh=None, batch_size=200, eta_bn_prop=1.0):

        # assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = {}
        topic_word = self.model.get_weights()
        results['topics'] = self.get_top_topics(topic_word, self.vocab, k=10)
        results['topic-document-matrix'] = self.get_doc_representaion(train_X, train_labels, train_prior_covars,
                                                                      train_topic_covars)
        results['topic-word-matrix'] = topic_word

        if test_X is not None:
            results['test-topic-document-matrix'] = self.get_doc_representaion(test_X, test_labels, test_prior_covars,
                                                                               test_topic_covars)

            self.model.eval()
            results['doc-losses'] = evaluate_perplexity(self.model, test_X, test_labels, test_prior_covars,
                                                        test_topic_covars, batch_size, eta_bn_prop=eta_bn_prop)

            results['word-losses'] = evaluate_word_perplexity(self.model, test_bow_fh, test_bow_sh,
                                                              test_labels, test_prior_covars, test_topic_covars,
                                                              batch_size, eta_bn_prop=eta_bn_prop)

        return results

    def get_doc_representaion(self, X, Y, PC, TC, batch_size=200):

        thetas = []
        n_items, _ = X.shape
        n_batches = int(np.ceil(n_items / batch_size))
        for i in range(n_batches):
            batch_xs, batch_ys, batch_pcs, batch_tcs = get_minibatch(X, Y, PC, TC, i, batch_size)
            thetas.append(self.model.compute_theta(batch_xs, batch_ys, batch_pcs, batch_tcs))
        theta = np.vstack(thetas)

        return theta.T

    def get_top_topics(self, beta, feature_names, k=10, sparsity_threshold=1e-5):
        topics = []
        for i in range(len(beta)):
            # sort the beta weights
            order = list(np.argsort(beta[i]))
            order.reverse()
            output = []
            # get the top words
            for j in range(len(order)):
                if len(output) < k:
                    if np.abs(beta[i][order[j]]) > sparsity_threshold:
                        output.append(feature_names[order[j]])
                else:
                    topics.append(output)
                    break
        return topics

    def train(self, network_architecture, X, Y, PC, TC, batch_size=200, training_epochs=100, display_step=10,
              X_dev=None, Y_dev=None,
              PC_dev=None, TC_dev=None, bn_anneal=True, init_eta_bn_prop=1.0, min_weights_sq=1e-7):
        # Train the model
        n_train, vocab_size = X.shape
        mb_gen = create_minibatch(X, Y, PC, TC, batch_size=batch_size, rng=self.rng)
        total_batch = int(n_train / batch_size)
        batches = 0
        eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon

        self.model.train()

        n_topics = network_architecture['n_topics']
        n_topic_covars = network_architecture['n_topic_covars']
        vocab_size = network_architecture['vocab_size']

        # create matrices to track the current estimates of the priors on the individual weights
        if network_architecture['l1_beta_reg'] > 0:
            l1_beta = 0.5 * np.ones([vocab_size, n_topics], dtype=np.float32) / float(n_train)
        else:
            l1_beta = None

        if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
            l1_beta_c = 0.5 * np.ones([vocab_size, n_topic_covars], dtype=np.float32) / float(n_train)
        else:
            l1_beta_c = None

        if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and \
            network_architecture['use_interactions']:
            l1_beta_ci = 0.5 * np.ones([vocab_size, n_topics * n_topic_covars], dtype=np.float32) / float(n_train)
        else:
            l1_beta_ci = None

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            accuracy = 0.
            avg_nl = 0.
            avg_kld = 0.
            # Loop over all batches
            for i in range(total_batch):
                # get a minibatch
                batch_xs, batch_ys, batch_pcs, batch_tcs = next(mb_gen)
                # do one minibatch update
                cost, recon_y, thetas, nl, kld = self.model.fit(batch_xs, batch_ys, batch_pcs, batch_tcs,
                                                                eta_bn_prop=eta_bn_prop, l1_beta=l1_beta,
                                                                l1_beta_c=l1_beta_c, l1_beta_ci=l1_beta_ci)

                # compute accuracy on minibatch
                if network_architecture['n_labels'] > 0:
                    accuracy += np.sum(np.argmax(recon_y, axis=1) == np.argmax(batch_ys, axis=1)) / float(n_train)

                # Compute average loss
                avg_cost += float(cost) / n_train * batch_size
                avg_nl += float(nl) / n_train * batch_size
                avg_kld += float(kld) / n_train * batch_size
                batches += 1
                if np.isnan(avg_cost):
                    print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                    print(
                        'Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    sys.exit()

            # if we're using regularization, update the priors on the individual weights
            if network_architecture['l1_beta_reg'] > 0:
                weights = self.model.get_weights().T
                weights_sq = weights ** 2
                # avoid infinite regularization
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l1_beta = 0.5 / weights_sq / float(n_train)

            if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
                weights = self.model.get_covar_weights().T
                weights_sq = weights ** 2
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l1_beta_c = 0.5 / weights_sq / float(n_train)

            if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and \
                network_architecture['use_interactions']:
                weights = self.model.get_covar_interaction_weights().T
                weights_sq = weights ** 2
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l1_beta_ci = 0.5 / weights_sq / float(n_train)

            # Display logs per epoch step
            if epoch % display_step == 0 and epoch > 0:
                if network_architecture['n_labels'] > 0:
                    print("Epoch:", '%d' % epoch, "; cost =", "{:.9f}".format(avg_cost),
                          "; training accuracy (noisy) =", "{:.9f}".format(accuracy))
                else:
                    print("Epoch:", '%d' % epoch, "cost=", "{:.9f}".format(avg_cost))

                # if X_dev is not None:
                #     # switch to eval mode for intermediate evaluation
                #     self.model.eval()
                #     dev_perplexity = evaluate_perplexity(self.model, X_dev, Y_dev, PC_dev, TC_dev, batch_size,
                #                                          eta_bn_prop=eta_bn_prop)
                #     n_dev, _ = X_dev.shape
                #     if network_architecture['n_labels'] > 0:
                #         dev_pred_probs = predict_label_probs(self.model, X_dev, PC_dev, TC_dev, eta_bn_prop=eta_bn_prop)
                #         dev_predictions = np.argmax(dev_pred_probs, axis=1)
                #         dev_accuracy = float(np.sum(dev_predictions == np.argmax(Y_dev, axis=1))) / float(n_dev)
                #         print("Epoch: %d; Dev perplexity = %0.4f; Dev accuracy = %0.4f" % (
                #             epoch, dev_perplexity, dev_accuracy))
                #     else:
                #         print("Epoch: %d; Dev perplexity = %0.4f" % (epoch, dev_perplexity))
                #     # switch back to training mode
                #     self.model.train()

            # anneal eta_bn_prop from 1.0 to 0.0 over training
            if bn_anneal:
                if eta_bn_prop > 0:
                    eta_bn_prop -= 1.0 / float(0.75 * training_epochs)
                    if eta_bn_prop < 0:
                        eta_bn_prop = 0.0

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

    def make_network(self, options, vocab_size, n_labels, n_prior_covars, n_topic_covars):
        # Assemble the network configuration parameters into a dictionary
        network_architecture = \
            dict(embedding_dim=options['embedding_dim'],
                 n_topics=options['num_topics'],
                 vocab_size=vocab_size,
                 n_labels=n_labels,
                 n_prior_covars=n_prior_covars,
                 n_topic_covars=n_topic_covars,
                 l1_beta_reg=options['l1_topics'],
                 l1_beta_c_reg=options['l1_topic_covars'],
                 l1_beta_ci_reg=options['l1_interactions'],
                 l2_prior_reg=options['l2_prior_covars'],
                 classifier_layers=1,
                 use_interactions=options['use_interactions'],
                 )
        return network_architecture

    def get_init_bg(self, vocab_freq):
        sums = vocab_freq + 1.0
        print("Computing background frequencies")
        print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
        bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
        return bg

    def split_data(self, data_list, train_indices, test_indices, dev_indices):
        train, dev, test = data_list, None, None

        if data_list is not None:
            # data_list = np.array(data_list)
            if train_indices is not None:
                train = data_list[train_indices]
            if dev_indices is not None:
                dev = data_list[dev_indices]
            if test_indices is not None:
                test = data_list[test_indices]

        return train, dev, test

    def split_bow(self, data):
        fh, sh = [], []
        if data is not None:
            fh = [' '.join([d[x] for x in range(0, len(d), 2)]) for d in data]
            sh = [' '.join([d[x] for x in range(1, len(d), 2)]) for d in data]
        return fh, sh


def create_minibatch(X, Y, PC, TC, batch_size=200, rng=None):
    # Yield a random minibatch
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            ixs = rng.randint(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)

        X_mb = np.array(X[ixs, :].todense()).astype('float32')
        if Y is not None:
            Y_mb = Y[ixs, :].astype('float32')
        else:
            Y_mb = None

        if PC is not None:
            PC_mb = PC[ixs, :].astype('float32')
        else:
            PC_mb = None

        if TC is not None:
            TC_mb = TC[ixs, :].astype('float32')
        else:
            TC_mb = None

        yield X_mb, Y_mb, PC_mb, TC_mb


def get_minibatch(X, Y, PC, TC, batch, batch_size=200):
    # Get a particular non-random segment of the data
    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / float(batch_size)))
    if batch < n_batches - 1:
        ixs = np.arange(batch * batch_size, (batch + 1) * batch_size)
    else:
        ixs = np.arange(batch * batch_size, n_items)

    X_mb = np.array(X[ixs, :].todense()).astype('float32')
    if Y is not None:
        Y_mb = Y[ixs, :].astype('float32')
    else:
        Y_mb = None

    if PC is not None:
        PC_mb = PC[ixs, :].astype('float32')
    else:
        PC_mb = None

    if TC is not None:
        TC_mb = TC[ixs, :].astype('float32')
    else:
        TC_mb = None

    return X_mb, Y_mb, PC_mb, TC_mb


def evaluate_perplexity(model, X, Y, PC, TC, batch_size, eta_bn_prop=0.0):
    # Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    n_items, vocab_size = X.shape
    doc_sums = np.array(X.sum(axis=1), dtype=float).reshape((n_items,))
    X = X.astype('float32')
    if Y is not None:
        Y = Y.astype('float32')
    if PC is not None:
        PC = PC.astype('float32')
    if TC is not None:
        TC = TC.astype('float32')
    losses = []

    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs = get_minibatch(X, Y, PC, TC, i, batch_size)
        batch_losses = model.get_losses(batch_xs, batch_ys, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop)
        losses.append(batch_losses)
    losses = np.hstack(losses)
    # perplexity = np.exp(np.mean(losses / doc_sums))
    return losses


def evaluate_word_perplexity(model, X_fh, X_sh, Y, PC, TC, batch_size, eta_bn_prop=0.0):
    # Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    n_items, vocab_size = X_sh.shape
    doc_sums = np.array(X_sh.sum(axis=1), dtype=float).reshape((n_items,))
    X_fh = X_fh.astype('float32')
    X_sh = X_sh.astype('float32')
    if Y is not None:
        Y = Y.astype('float32')
    if PC is not None:
        PC = PC.astype('float32')
    if TC is not None:
        TC = TC.astype('float32')
    losses = None

    n_items, _ = X_sh.shape
    n_batches = int(np.ceil(n_items / batch_size))
    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs = get_minibatch(X_sh, Y, PC, TC, i, batch_size)
        batch_xs_train, _, _, _ = get_minibatch(X_fh, Y, PC, TC, i, batch_size)
        batch_xs_train = torch.from_numpy(np.array(batch_xs_train)).float().to(model.device)
        recon = model.get_reconstruction(batch_xs, batch_ys, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop)
        loss = -(batch_xs_train * (recon + 1e-10).log()).sum(1)
        loss = loss.cpu().detach().numpy()

        if losses is None:
            losses = loss
        else:
            losses = np.concatenate((losses, loss), axis=0)

    return losses


def predict_label_probs(model, X, PC, TC, batch_size=200, eta_bn_prop=0.0):
    # Predict a probability distribution over labels for each instance using the classifier part of the network

    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    pred_probs_all = []

    # make predictions on minibatches and then combine
    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs = get_minibatch(X, None, PC, TC, i, batch_size)
        Z, pred_probs = model.predict(batch_xs, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop)
        pred_probs_all.append(pred_probs)

    pred_probs = np.vstack(pred_probs_all)

    return pred_probs
