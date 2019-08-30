# intent_ner

This repo contains the solutions for the two tasks 

1. Intent classification
2. NER

Note: Except for the Large NER model I've uploaded every other models. if in case you need to try it.

You can find the various analysis, explorations of the data in the EDA notebooks I've attached and
you can also find the various modelling approaches I've tried in other notebooks. 

### High level overview of the tasks: 

1. On the Intent classification task, we can reach an accuracy of 99% with a good precision and recall score as well.
2. On the NER task, we build on top of the small & large versions of the spacy `en` model and I've also augmented the data
to increase our accuracy(went up by around 20%).
3. data augmentation approach: scrapped a list of german cities from wikipedia, some vehicles name from a website and 
populated it in our base template sentences which were derived from our train & test data.


