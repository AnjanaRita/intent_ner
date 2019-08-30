import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from collections import namedtuple


def json_2_dataframe(filename):
    with open(filename, 'rb') as datafile:
        dataset = json.load(datafile)
    return pd.DataFrame(dataset.get('sentences'))


def train_test_spliter(data):
    train_test_split = namedtuple('train_test_split', ['train', 'test'])
    train = shuffle(data[data["training"] == True], random_state=42)
    test = shuffle(data[data["training"] == False], random_state=42)
    return train_test_split(train, test)


def corpus_entity_info(train_ner, test_ner, col1="train", col2="test"):
    train_stat = train_ner.groupby('entity')['text'].agg(
        ['count']).rename(columns={'count': col1})
    test_stat = test_ner.groupby('entity')['text'].agg(
        ['count']).rename(columns={'count': col2})
    train_test_stat = pd.merge(
        train_stat,
        test_stat,
        on='entity',
        how='left').replace(
        np.nan,
        0)
    return train_test_stat.astype(int)


def spacy_data_conversion(dataset, mode="train"):
    formated_data = []
    for text, ent in zip(dataset.text.values, dataset.entities.values):
        entitiy_list = [(text.find(i.get('text')),
                         text.find(i.get('text')) + len(i.get("text")),
                         i.get('entity')) for i in ent]
        if mode == "train":
            ents = {}
            ents['entities'] = entitiy_list
        else:
            ents = entitiy_list
        formated_data.append((text, ents))
    return formated_data
