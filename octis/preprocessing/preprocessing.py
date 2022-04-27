import re
import string
from typing import List, Union

import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map  # or thread_map
from pathlib import Path
from octis.dataset.dataset import Dataset
from collections import Counter
import pandas as pd

import utils.file_handling as fh

"""
Maps the language to its corresponding spacy model
"""
spacy_model_mapping = {'chinese': 'zh_core_web_sm', 'danish': 'nl_core_news_sm', 'dutch': 'nl_core_news_sm',
                       'english': 'en_core_web_sm', 'french': 'fr_core_news_sm', 'german': 'de_core_news_sm',
                       'greek': 'el_core_news_sm', 'italian': 'it_core_news_sm', 'japanese': 'ja_core_news_sm',
                       'lithuanian': 'lt_core_news_sm', 'norwegian': 'nb_core_news_sm', 'polish': 'pl_core_news_sm',
                       'portuguese': 'pt_core_news_sm', 'romanian': 'ro_core_news_sm', 'russian': 'ru_core_news_sm',
                       'spanish': 'es_core_news_sm'}
UNKNOWN_TOKEN = "<UNK>"


class Preprocessing:
    def __init__(self, lowercase: bool = True, vocabulary: List[str] = None, max_features: int = None,
                 min_df: float = 0.0, max_df: float = 1.0, remove_punctuation: bool = True,
                 punctuation: str = string.punctuation, remove_numbers: bool = True,
                 lemmatize: bool = True, stopword_list: Union[str, List[str]] = None, min_chars: int = 1,
                 min_words_docs: int = 0, language: str = 'english', split: bool = True, verbose: bool = False,
                 num_processes: int = None, save_original_indexes=True, remove_stopwords_spacy: bool = True,
                 labels: List[str] = None, covariates: List[str] = None,
                 min_label_count: int = None, min_covars_count: int = None,
                 max_label_count: int = None, max_covars_count: int = None,
                 remove_alphanum: bool = True, strip_html: bool = True, ):
        """
        init Preprocessing

        :param lowercase: if true, words in documents are reduced to lowercase (default: true)
        :type lowercase: boolean
        :param vocabulary: the vocabulary of the corpus to preprocess (default: None)
        :type vocabulary: list
        :param max_features: maximum number of words that the vocabulary must contain. The less frequent
        words will be removed. If it's not None, then max_df and min_df are ignored (default: None)
        :type max_features: int
        :param min_df: words below this minumum document frequency will be removed (default: 0.0)
        :type min_df: float
        :param max_df: words above this maximum document frequency will be removed (default: 1.0)
        :type max_df: float
        :param remove_punctuation: if true, punctuation will be removed (default: true)
        :type remove_punctuation: bool
        :param punctuation: string containing all the punctuation chars that need to be removed (default:
        string.punctuation)
        :type punctuation: str
        :param remove_numbers: if true, numbers will be removed
        :type remove_numbers: bool
        :param remove_stopwords_spacy: bool , if true use spacy to remove stopwords (default: true)
        :param lemmatize: if true, words will be lemmatized using a spacy model according to the language that has been
        set (default: true)
        :type lemmatize: bool
        :param stopword_list: if a list of strings is passed, the strings will be removed from the texts. Otherwise,
        if a str is passed, it represents the language of the stopwords that need to be removed. The stopwords are
        spacy's stopwords (default: None)
        :type stopword_list: str or list of str
        :param min_chars: mininum number of characters that a token should have (default: 1)
        :type min_chars: int
        :param min_words_docs: minimun number of words that a document should contain (default: 0)
        :type min_words_docs: int
        :param language: language of the documents. It needs to be set for the lemmatizer (default: english)
        :type language: str
        :param split: if true, the corpus will be split in train (85%), testing (7.5%) and validation (7.5%) set (
        default: true)
        :type split: bool
        :param verbose: if true, some steps of the preprocessing will be printed on screen (default: false)
        :type verbose: bool
        :param num_processes: number of processes to run the preprocessing
        :type num_processes: int
        :param labels: list of labels associated with each document (optional, comma-separated)
        :param covariates: list of covariates associated with each document (optional, comma-separated)
        :param min_label_count: int , exclude labels that occur in less than this number of documents
        :param max_label_count: int, the max number of labels to be kept based on their document frequency
        :param min_covars_count: int, exclude covariates that occur in less than this number of documents
        :param max_covars_count:int, the max number of covariates to be kept based on their document frequency
        :param remove_alphanum: bool, if true remove tokens made of a mixture of letters and numbers
        :param strip_html: bool, if true, Strip HTML tags
        """

        self.vocabulary = vocabulary
        self.lowercase = lowercase
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.remove_punctuation = remove_punctuation
        self.punctuation = punctuation
        self.lemmatize = lemmatize
        self.language = language
        self.num_processes = num_processes
        self.remove_numbers = remove_numbers
        self.save_original_indexes = save_original_indexes

        if self.lemmatize:
            lang = spacy_model_mapping[self.language]
            try:
                self.spacy_model = spacy.load(lang)
            except IOError:
                raise IOError("Can't find model " + lang + ". Check the data directory or download it using the "
                                                           "following command:\npython -m spacy download " + lang)
        self.split = split
        self.verbose = verbose
        self.remove_stopwords_spacy = remove_stopwords_spacy

        stopwords = []
        # if stopwords is None then stopwords are not removed
        if stopword_list is None:
            self.remove_stopwords_spacy = False
        else:
            # if custom list is specified, then we do not use spacy stopwords
            if type(stopword_list) == list:
                stopwords = set(stopword_list)
                self.remove_stopwords_spacy = False
            elif self.remove_stopwords_spacy:
                assert stopword_list == language
            else:
                # if remove_stopwords_spacy is false, then use MALLET English stopwords
                if 'english' in stopword_list:
                    stop_word_path = Path(__file__).parent.joinpath('stopwords', 'english.txt')
                    with open(stop_word_path) as fr:
                        stopwords = [line.strip() for line in fr.readlines()]
                        assert stopword_list == language

        self.stopwords = stopwords
        self.min_chars = min_chars
        self.min_doc_words = min_words_docs
        self.preprocessing_steps = []

        self.labels = labels
        self.covariates = covariates
        self.min_label_count = min_label_count
        self.max_label_count = max_label_count
        self.min_covars_count = min_covars_count
        self.max_covars_count = max_covars_count
        self.remove_alphanum = remove_alphanum
        self.strip_html = strip_html

    def preprocess_dataset(self, documents=None, documents_path=None):
        """
        preprocess the input dataset

        :param documents_path: path to the documents file. Each row of the file represents a document
        :type documents_path: str

        :return octis.dataset.dataset.Dataset
        """
        assert documents is None and documents_path is None, 'You should either pass a document set or path to dataset'
        if documents is None:
            documents = pd.DataFrame.from_records(fh.read_jsonlist(documents_path))

        ids, docs, labels, covariates = documents['id'].tolist(), documents['text'].tolist(), None, None

        # Read Labels and Covariates
        if self.labels:
            labels = self.read_labels_list(documents, self.labels)
        if self.covariates:
            covariates = self.read_labels_list(documents, self.covariates)

        # Filter Labels and Covariates based on frequency limits of labels/covariates
        if labels:
            docs, labels, filtered_ids = self.filter_labels(labels, docs, ids,
                                                            self.min_label_count,
                                                            self.max_label_count,
                                                            filter_none=True)
            if covariates:
                ids, covariates = zip(*[(x, y) for x, y in zip(ids, covariates) if x in filtered_ids])
                ids, covariates = list(ids), list(covariates)
            else:
                ids = filtered_ids

        if covariates:
            _, covariates, _ = self.filter_labels(covariates, docs, ids,
                                                  self.min_covars_count,
                                                  self.max_covars_count,
                                                  filter_none=False)
        # Clean text
        if self.num_processes is not None:
            docs = process_map(self.simple_preprocessing_steps, docs, max_workers=self.num_processes, chunksize=1)
        else:
            docs = self.simple_preprocessing_steps(docs)

        # Filter Vocabulary
        vocabulary = self.filter_words(docs)
        vocab = set(vocabulary)
        print(f"created vocab: {len(vocab)}")
        docs = [' '.join([w for w in doc.split() if w in vocab]) for doc in docs]
        ids = [j for i, j in enumerate(ids) if len(docs[i]) > self.min_doc_words]
        if labels:
            labels = [j for i, j in enumerate(labels) if len(docs[i]) > self.min_doc_words]
        if covariates:
            covariates = [j for i, j in enumerate(covariates) if len(docs[i]) > self.min_doc_words]
        docs = [doc for doc in docs if len(doc) > self.min_doc_words]

        if self.verbose:
            print("words filtering done")
        metadata = {"total_documents": len(docs), "vocabulary_length": len(vocabulary),
                    "preprocessing-info": self.preprocessing_steps}
        if self.split:
            partitioned_docs = None
            partitioned_covariates = None
            partitioned_labels = None

            train, test = train_test_split(range(len(docs)), test_size=0.15, random_state=1)
            train, validation = train_test_split(train, test_size=3 / 17, random_state=1)

            metadata["last-training-doc"] = len(train)
            metadata["last-validation-doc"] = len(validation) + len(train)
            partitioned_docs = [docs[doc].split() for doc in train + validation + test]
            document_indexes = [ids[doc] for doc in train + validation + test]

            if labels:
                partitioned_labels = [labels[doc] for doc in train + validation + test]

            if covariates:
                partitioned_covariates = [covariates[doc] for doc in train + validation + test]

            unprocessed_docs = documents[documents['id'].isin(document_indexes)]['text'].tolist()
            if self.save_original_indexes:
                return Dataset(partitioned_docs, raw_corpus=unprocessed_docs, vocabulary=vocabulary, metadata=metadata,
                               labels=partitioned_labels,
                               covariates=partitioned_covariates, document_indexes=document_indexes)
            else:
                return Dataset(partitioned_docs, raw_corpus=unprocessed_docs, vocabulary=vocabulary, metadata=metadata,
                               labels=partitioned_labels,
                               covariates=partitioned_covariates)
        else:
            docs = [doc.split() for doc in docs]
            unprocessed_docs = documents[documents['id'].isin(ids)]['text'].tolist()
            if self.save_original_indexes:
                return Dataset(docs, raw_corpus=unprocessed_docs, vocabulary=vocabulary, metadata=metadata,
                               labels=labels, covariates=covariates,
                               document_indexes=ids)
            else:

                return Dataset(docs, raw_corpus=unprocessed_docs, vocabulary=vocabulary, metadata=metadata,
                               labels=labels, covariates=covariates)

    # TODO:: refactor
    def read_labels_list(self, documents, labels_names):
        """
        Read label fields in documents and combine them into list. Each document is associated with a list of labels
        :param documents: Pandas Dataframe, the set of documents. Each row in the dataframe represents a document
        :param labels_names: list of label fields to be extracted from documents.
        :return: List of label corresponding to document
        """
        labels_list = {}
        if type(labels_names) == str:
            if ',' in labels_names:
                labels_names = labels_names.split(',')
            else:
                labels_names = [labels_names]

        for l in labels_names:
            if l in documents.columns:
                documents[l].fillna('', inplace=True)
                labels = documents[l].tolist()
                updated_labels = []
                for x in labels:
                    if type(x) == list:
                        updated_labels.append([str(y) for y in x])
                    elif type(x) == str and x == '':
                        updated_labels.append([])
                    else:
                        updated_labels.append([str(x)])
                labels_list[l] = updated_labels
        # x={l:[y for y in documents[l].tolist()] for l in labels_names if l in documents.columns}
        all_labels = list(map(list, zip(*[labels_list[k] for k in sorted(labels_list)])))
        final_labels = []
        for labels in all_labels:
            temp_labels = []
            for l in labels:
                temp_labels.extend(l)
            temp_labels = [str(l).lower() for l in temp_labels if l]
            final_labels.append(temp_labels)
        return final_labels

    # TODO: spilt this into two methods. Take out document filtering
    def filter_labels(self, labels, docs, ids, min_label_count=None, max_label_count=None, filter_none=False):
        """
        Calculate labels counts in documents, then filter labels that do not meet the passed frequency criteria.
        :param labels: List of labels associated with documents. Each entry is a list of labels ( multilables)
        :param docs: List of documents
        :param ids: List of document ids
        :param min_label_count: int, exclude labels that occur in less than this number of documents
        :param max_label_count: int, the max number of labels to be kept based on their document frequency
        :param filter_none: bool, if True, remove documents with empty label list
        :return: List of documents, list of labels, list of document ids
        """
        final_docs, final_labels, document_indexes = [], [], []
        labels_to_remove = set()
        if min_label_count:
            all_labels = [x.lower() for y in labels for x in y]
            labels_to_remove = set([k for k, v in dict(Counter(all_labels)).items() if v <= min_label_count])

        elif max_label_count:
            label_vectorizer = CountVectorizer(max_features=max_label_count, stop_words=self.stopwords)
            all_labels = [x.lower() for y in labels for x in y]
            label_vectorizer.fit_transform(all_labels)
            u_labels = set(label_vectorizer.get_feature_names())
            labels_to_remove = list(set(all_labels).difference(set(u_labels)))

        if len(labels_to_remove) > 0:
            for i, doc, label in zip(ids, docs, labels):
                doc_final_labels = []
                for l in label:
                    if l not in labels_to_remove:
                        doc_final_labels.append(l)
                # if doc_final_labels:
                if filter_none and not doc_final_labels:
                    pass
                else:
                    final_docs.append(doc)
                    final_labels.append(doc_final_labels)
                    document_indexes.append(i)
        else:
            final_docs = docs
            final_labels = labels
            document_indexes = ids
        return final_docs, final_labels, document_indexes

    def filter_words(self, docs):
        """
        Select the vocabulary list in documents
        :param docs:  List of documents
        :return: Generated vocabulary list
        """
        if self.vocabulary is not None:
            self.preprocessing_steps.append('filter words by vocabulary')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = CountVectorizer(max_df=self.max_df, min_df=self.min_df, vocabulary=self.vocabulary,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + ",}\b",
                                         lowercase=self.lowercase, stop_words=self.stopwords)

        elif self.max_features is not None:
            self.preprocessing_steps.append('filter vocabulary to ' + str(self.max_features) + ' terms')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            # we ignore df_max_freq e df_min_freq because self.max_features is not None
            vectorizer = CountVectorizer(lowercase=self.lowercase, max_features=self.max_features,
                                         stop_words=self.stopwords,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b")

        else:

            # string.punctuation

            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = CountVectorizer(max_df=self.max_df, min_df=self.min_df, lowercase=self.lowercase,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b",
                                         stop_words=self.stopwords)

        vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names()
        return vocabulary

    def clean_text(self, text):
        text = text.replace("\\", " ")
        text = text.replace("/", " ")
        text = text.replace("-", " ")
        text = text.replace("–", " ")
        text = text.replace('\n', '')
        text = text.replace('\t', '')
        text = text.translate(str.maketrans('', '', string.punctuation)).strip()
        replace_by_space_re = re.compile('[\n\"\'`/(){}\[\]\|@,;#]')
        text = re.sub(replace_by_space_re, ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()
        return text

    def simple_preprocessing_steps(self, docs):
        tmp_docs = []
        for d in docs:
            new_d = d
            if self.strip_html:
                new_d = re.sub(r'<[^>]+>', '', new_d)
            else:
                # replace angle brackets
                new_d = re.sub(r'<', '(', new_d)
                new_d = re.sub(r'>', ')', new_d)

            if self.lowercase:
                new_d = new_d.lower()

            if self.lemmatize:
                if self.remove_stopwords_spacy:
                    new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d) if not token.is_stop])
                elif self.stopwords:
                    new_d = ' '.join(
                        [token.lemma_ for token in self.spacy_model(new_d) if token.lemma_ not in set(self.stopwords)])
                else:
                    new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d)])
            else:
                if self.stopwords:
                    new_d = ' '.join([w for w in new_d.split() if len(w) > 2 and w not in self.stopwords])

            if self.remove_punctuation:
                new_d = new_d.translate(str.maketrans(self.punctuation, ' ' * len(self.punctuation)))

            if self.remove_numbers:
                new_d = new_d.translate(str.maketrans("0123456789", ' ' * len("0123456789")))

            new_d = self.clean_text(new_d)
            new_d = " ".join(new_d.split())
            tmp_docs.append(new_d)
        return tmp_docs
