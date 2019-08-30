import spacy
import random
from spacy.util import minibatch, compounding


def custom_train(dataset, new_labels, model="en",
                 new_model_name="chatbot", output_dir=None, n_iter=20, verbose=False):
    """
    Set up the pipeline and entity recognizer, and train the new entity.
    """

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe("ner")

    for label in new_labels:
        ner.add_label(label)

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 8.0, 1.001)
        for itn in range(n_iter):
            random.shuffle(dataset)
            batches = minibatch(dataset, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer,
                           drop=0.35, losses=losses)
            if verbose:
                print(f"Losses after {itn+1} iteration {losses} ")

    print('model is trained')
    if output_dir:
        nlp.to_disk(output_dir)
        print(f"model is saved to {output_dir}")
    return nlp
