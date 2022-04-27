import argparse
import os
import string

import time
from collections import defaultdict
from datetime import datetime
import psutil as psutil

from octis.models.CTM import CTM
from octis.models.ProdLDA import ProdLDA
from octis.models.Scholar import scholar
from octis.preprocessing.preprocessing import Preprocessing
from evaluate import evaluate_metric
from utils import utils

process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser(usage="Topic Modelling")
parser.add_argument('--model-type', dest="model_type", default='Scholar', required=True,
                    help='The type of the model to be trained [ProdLDA|Scholar|CTM|SuperCTM] : default=%default')
parser.add_argument('--num-topics', dest="num_topics", default=600, type=int, required=False,
                    help='The number of topics to be trained: default=%default')
parser.add_argument('--epochs', dest="epochs", default=100, type=int, required=False,
                    help='Number of training epochs: default=%default')
parser.add_argument('--runs', dest="runs", default=5, type=int, required=False,
                    help='Number of model runs: default=%default')
parser.add_argument('--use-partitions', dest="use_partitions", default=False, type=bool, required=False,
                    help='if true the model will be trained on the training set and evaluated on the test')
parser.add_argument('--dataset-path', dest="dataset_path", required=True,
                    help='Path to training dataset: default=%default')
parser.add_argument('--training-embeddings', dest="training_embeddings", default=None, required=False,
                    help='Path to embedded training dataset: default=%default')
parser.add_argument('--testing-embeddings', dest="testing_embeddings", default=None, required=False,
                    help='Path to embedded test dataset: default=%default')
parser.add_argument('--save-dir', dest="save_dir", required=False,
                    help='Directory to save model checkpoints: default=%default')
parser.add_argument('--model-output', dest="model_output", required=False, default='output',
                    help='Directory to save model output : default=%default')

# Text preprocessing:
parser.add_argument('--max-features', dest="max_features", type=int, default=13000, required=False,
                    help='Size of the vocabulary (by most common): default=%default')
parser.add_argument('--label', dest='label', default=None,
                    help='field(s) to use as label (comma-separated): default=%default')
parser.add_argument('--covariate', dest='covariate', default=None,
                    help='field(s) to use as covariates (comma-separated): default=%default')
parser.add_argument('--interactions', dest="interactions", type=bool, default=False,
                    help='Use interactions between topics and topic covariates: default=%default')
parser.add_argument('--min-label-count', dest="min_label_count", type=int, default=None, required=False,
                    help='Exclude labels that occur in less than this number of documents: default=%default')
parser.add_argument('--max-label-count', dest="max_label_count", type=int, default=None, required=False,
                    help='The top number of labels to keep: default=%default')
parser.add_argument('--min-covars-count', dest="min_covars_count", type=int, default=None, required=False,
                    help='The top number of labels to keep: default=%default')
parser.add_argument('--max-covars-count', dest="max_covars_count", type=int, default=None, required=False,
                    help='The top number of labels to keep: default=%default')

parser.add_argument('--stopwords', dest='stopwords', default='english',
                    help='List of stopwords to exclude [None|mallet|snowball]: default=%default')
parser.add_argument('--lemmatize', dest='lemmatize', type=bool, default=False, required=False,
                    help='Lemmatized words using a spacy model according to the language that has been: default=%default')
parser.add_argument('--min-doc-count', dest='min_doc_count', default=0,
                    help='Exclude words that occur in less than this number of documents')
parser.add_argument('--min-doc-freq', dest='min_df', default=0.0,
                    help='Exclude words that occur in more than this proportion of documents')
parser.add_argument('--max-doc-freq', dest='max_df', default=1.0,
                    help='Exclude words that occur in more than this proportion of documents')
parser.add_argument('--remove-num', action="store_true", dest="remove_num", default=True,
                    help='remove tokens made of only numbers: default=%default')
parser.add_argument('--remove-alphanum', action="store_true", dest="remove_alphanum", default=True,
                    help="Remove tokens made of a mixture of letters and numbers: default=%default")
parser.add_argument('--strip-html', action="store_true", dest="strip_html", default=True,
                    help='Strip HTML tags: default=%default')
parser.add_argument('--lower', action="store_true", dest="lower", default=True,
                    help='Do not lowercase text: default=%default')
parser.add_argument('--min_chars', dest='min_chars', default=2,
                    help='Minimum token length: default=%default')

models = {
    'ProdLDA': ProdLDA,
    'Scholar': scholar,
    'CTM': CTM,
    'SuperCTM': CTM
}


def run_topic_modelling(TMmodel, dataset, runs=5, evals=[], parameters={}, model_output_dir=None):
    run_time = []
    evaluation_values = defaultdict(list)

    for r in range(runs):
        print('----------------------------')
        print(f'        Run # {r}')
        print('----------------------------')
        start_time = time.time()
        model = TMmodel(**parameters)
        print("Starting Training. Current Time =", datetime.now().strftime("%H:%M:%S"))
        model_output = model.train_model(dataset=dataset)
        utils.save_model_output(model_output, r, model_output_dir)
        # Temp Code
        utils.save_document_embed(model_output, dataset.get_document_indexes(), r, model_output_dir)
        del model
        print("Finished Training in {}".format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

        print("\nStarting Evaluation. Current Time =", datetime.now().strftime("%H:%M:%S"))
        start_time = time.time()
        for eval in evals:
            s = datetime.now()
            score = evaluate_metric(eval, model_output, dataset)
            evaluation_values[eval].append(score)
            print("eval_type {}\tscore {}\tTime {}\tMemory {:.3f} GB".format(eval, score, datetime.now() - s,
                                                                             process.memory_info().rss / (1024 ** 3)))
        print("\nFinished Evaluation in {}".format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

        end_time = time.time()
        run_time.append(end_time - start_time)
        print("Total Time is %s" % (time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))))

    print('\n')
    for eval in evals:
        print(f'Average {eval} = {sum(evaluation_values[eval]) / len(evaluation_values[eval])}')
    print("Average Time per Run is %s\n\n" % (time.strftime('%H:%M:%S', time.gmtime(sum(run_time) / len(run_time)))))


def main(args):
    print("Get Configurations")
    configs = parser.parse_args(args)

    print("Reading and Preprocessing Training Dataset")
    preprocessor = Preprocessing(vocabulary=None, max_features=configs.max_features,
                                 labels=configs.label, min_label_count=configs.min_label_count,
                                 max_label_count=configs.max_label_count,
                                 covariates=configs.covariate, min_covars_count=configs.min_covars_count,
                                 max_covars_count=configs.max_covars_count,
                                 min_df=configs.min_df, max_df=configs.max_df,
                                 min_words_docs=configs.min_doc_count, min_chars=configs.min_chars,
                                 remove_punctuation=True,
                                 remove_numbers=configs.remove_num,
                                 remove_alphanum=configs.remove_alphanum,
                                 strip_html=configs.strip_html,
                                 lowercase=configs.lower,
                                 punctuation=string.punctuation,
                                 lemmatize=configs.lemmatize,
                                 stopword_list=configs.stopwords,
                                 split=configs.use_partitions,
                                 language='english', verbose=False, num_processes=None,
                                 save_original_indexes=True, remove_stopwords_spacy=False)

    dataset = preprocessor.preprocess_dataset(documents_path=configs.dataset_path)

    print(f"Create Model {configs.model_type}")
    parameters = utils.create_params(configs.model_type, configs)
    model = models[configs.model_type]

    print("Start Training and Evaluation Pipline")
    evals = ['NPMI', 'C_V', 'U_MASS', 'C_UCI']

    print('Training Model {} with {} Topics'.format(configs.model_type, configs.num_topics))
    model_output_dir = f'{configs.model_output}/{configs.model_type}_w_{configs.max_features}' \
                       f'_t_{configs.num_topics}_{time.strftime("%Y%m%d%H%M")}'
    run_topic_modelling(TMmodel=model,
                        dataset=dataset,
                        runs=configs.runs,
                        evals=evals,
                        parameters=parameters,
                        model_output_dir=model_output_dir)
