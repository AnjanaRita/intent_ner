import random
import requests
import numpy as np
import pandas as pd
from lxml import html

city_data = requests.get(
    "https://en.wikipedia.org/wiki/List_of_cities_and_towns_in_Germany")
parser = html.fromstring(city_data.content)
city_name = parser.xpath('//table//ul//li/a/@title')

car_data = requests.get("https://www.carlogos.org/All-List/")
car_parser = html.fromstring(car_data.content)
vehicle_names = car_parser.xpath(
    './/div[@class="main-l"]//dl[@class="alllist"]/dd/a/text()')

line = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8']

entity_mapping = {'StationDest': city_name,
                  'Vehicle': vehicle_names,
                  'Line': line,
                  'StationStart': city_name}


def fake_data_generation(dataset: 'pd.DataFrame',
                         training: 'str' = True) -> 'pd.DataFrame':
    ner_fake_data = []
    for count, row in dataset.iterrows():
        data = row.get('text')
        ent = row.get('entities')
        intent = row.get('intent')
        for sample_data in range(0, np.random.randint(3, 10)):
            ent_list = []
            temp_text = data
            for _ent in ent:
                temp = _ent.copy()
                old_text = temp['text']
                ent_name = temp['entity']

                if ent_name in list(entity_mapping.keys()):
                    new_text = random.choice(entity_mapping[ent_name])
                    temp_text = temp_text.replace(old_text, new_text)
                else:
                    new_text = old_text
                d = {}
                d['text'] = new_text
                d['entity'] = ent_name
                d['start'] = temp['start']
                d['stop'] = temp['start'] + len(new_text.split(' '))
                ent_list.append(d)
            ner_fake_data.append({"text": temp_text,
                                  "entities": ent_list,
                                  "intent": intent, "training": training})
    return pd.DataFrame(ner_fake_data)
