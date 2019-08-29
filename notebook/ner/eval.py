import pandas as pd
from spacy.gold import GoldParse
from spacy.scorer import Scorer

def ner_model_evalution(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        try:
            doc_gold_text = ner_model.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=annot)
            pred_value = ner_model(input_)
            scorer.score(pred_value, gold)
        except Exception as ex:
            print(input_, annot)
    temp_score =  scorer.scores
    overall ={"overall" : {"p" : temp_score.get('ents_p'),
                       "r" : temp_score.get('ents_r'),
                       "f" : temp_score.get('ents_f')}}
    overall.update(temp_score.get("ents_per_type"))
    eval_frame = pd.DataFrame(overall).reset_index().replace({"f":"f1_score","p":"precision_score", "r":"recall_score"})
    eval_frame = eval_frame.set_index('index').T
    return eval_frame