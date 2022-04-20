"""A script to implement the functions to load the configurations"""
import ast


def load_configs(config_file):
    """
    Load the training configurations
    :param config_file:
    :return: a dict with the training arguments
    """
    arg_dict = {}
    files = config_file.get('data')
    if isinstance(files['max_features'], int):
        arg_dict['max_features'] = files['max_features']
    else:
        raise ValueError('max_features must be an int')

    if isinstance(files['max_labels'], int):
        arg_dict['max_labels'] = files['max_labels']
    else:
        raise ValueError('max_labels must be an int')

    parameters = config_file.get('parameters')
    if isinstance(parameters['model_type'], str):
        arg_dict['model_type'] = parameters['model_type']
    else:
        raise ValueError('model_type must be a str')
    if isinstance(parameters['epochs'], int):
        arg_dict['epochs'] = parameters['epochs']
    else:
        raise ValueError('epochs must be an integer')
    if isinstance(parameters['num_topics'], list):
        arg_dict['num_topics'] = parameters['num_topics']
    else:
        raise ValueError('num_topics must be list of integers')
    if isinstance(parameters['runs'], int):
        arg_dict['runs'] = parameters['runs']
    else:
        raise ValueError('runs must be and integer')

    bert_model = config_file.get('bert_model')
    if isinstance(bert_model['training'], str):
        arg_dict['training'] = bert_model['training']
    else:
        raise ValueError('training must be an str')

    if isinstance(bert_model['testing'], str):
        arg_dict['testing'] = bert_model['testing']
    else:
        raise ValueError('testing must be an str')

    input = config_file.get('input')
    if isinstance(input['dataset_path'], str):
        arg_dict['dataset_path'] = input['dataset_path']
    else:
        raise ValueError('dataset_path must be an str')

    output = config_file.get('output')
    if isinstance(output['save_path'], str):
        arg_dict['save_path'] = output['save_path']
    else:
        raise ValueError('save_path must be an str')
    if isinstance(output['model_output'], str):
        arg_dict['model_output'] = output['model_output']
    else:
        raise ValueError('model_output must be an str')
    return arg_dict


def update_configs(configs, args):
    if args.model_type:
        configs['model_type'] = args.model_type

    if args.num_topics:
        try:
            configs['num_topics'] = ast.literal_eval(args.num_topics)
        except:
            raise ValueError('num_topics must be an int')

    if args.epochs:
        try:
            configs['epochs'] = int(args.epochs)
        except:
            raise ValueError('epochs must be an int')

    if args.runs:
        try:
            configs['runs'] = int(args.runs)
        except:
            raise ValueError('runs must be an int')

    if args.dataset_path:
        configs['dataset_path'] = args.dataset_path

    if args.max_features:
        try:
            configs['max_features'] = int(args.max_features)
        except:
            raise ValueError('max_features must be an int')

    if args.max_labels:
        try:
            configs['max_labels'] = int(args.max_labels)
        except:
            raise ValueError('max_labels must be an str')

    if args.training:
        configs['training'] = args.training

    if args.testing:
        configs['testing'] = args.testing

    if args.save_path:
        configs['save_path'] = args.save_path

    if args.model_output:
        configs['model_output'] = args.model_output

    return configs

# if args.model_type:
#     if isinstance(args.model_type, str):
#         configs['model_type'] =args.model_type
#     else:
#         raise ValueError('model_output must be an str')
#
# if args.num_topics:
#     if isinstance(args.num_topics, int):
#         configs['num_topics'] =args.num_topics
#     else:
#         raise ValueError('num_topics must be an int')
#
# if args.epochs:
#     if isinstance(args.epochs, int):
#         configs['epochs'] =args.epochs
#     else:
#         raise ValueError('epochs must be an int')
#
# if args.runs:
#     if isinstance(args.runs, int):
#         configs['runs'] =args.runs
#     else:
#         raise ValueError('runs must be an int')
#
# if args.dataset_path:
#     if isinstance(args.dataset_path, str):
#         configs['dataset_path'] =args.dataset_path
#     else:
#         raise ValueError('dataset_path must be an str')
#
# if args.max_features:
#     if isinstance(args.max_features, int):
#         configs['max_features'] =args.max_features
#     else:
#         raise ValueError('max_features must be an int')
#
# if args.max_labels:
#     if isinstance(args.max_labels, str):
#         configs['max_labels'] =args.max_labels
#     else:
#         raise ValueError('max_labels must be an str')
#
# if args.training:
#     if isinstance(args.training, str):
#         configs['training'] =args.training
#     else:
#         raise ValueError('training must be an str')
#
# if args.testing:
#     if isinstance(args.testing, str):
#         configs['testing'] =args.testing
#     else:
#         raise ValueError('testing must be an str')
#
# if args.save_path:
#     if isinstance(args.save_path, str):
#         configs['save_path'] =args.save_path
#     else:
#         raise ValueError('save_path must be an str')
#
# if args.model_output:
#     if isinstance(args.model_output, str):
#         configs['model_output'] =args.model_output
#     else:
#         raise ValueError('model_output must be an str')
