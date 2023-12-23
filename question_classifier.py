import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def transfer_relative_path(relative_path):
    current_working_directory = os.getcwd()
    # transfer relative path to absolute path.
    absolute_path = os.path.join(current_working_directory, relative_path)
    return absolute_path


def extract_question_sentence(doc):
    if len(doc.sentences) == 1:
        return doc.sentences[0]

    for sent in doc.sentences:
        if sent.tokens[-1].text != '.':
            return sent


def classify_question(text):
    # 0: open question; 1: boolean question
    tokenizer = AutoTokenizer.from_pretrained("PrimeQA/tydiqa-boolean-question-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("PrimeQA/tydiqa-boolean-question-classifier")

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    question_type = outputs.logits.argmax(dim=-1).item()

    return question_type


def classify_entity_question(question):
    # PERSON (29, 30, 31)
    # NORP (Nationalities/religious/political groups) NORP （国籍/宗教/政治团体）(16)
    # ORG (Organization) ORG （组织）(28)
    # GPE (Countries/cities/states) GPE （国家/城市/州）(32,33,36)
    # LOC (Location) LOC （位置）(34, 35)
    # FAC, PRODUCT, WORK_OF_ART, LAW(2,3,5,9,10,14,15,22)
    # EVENT (8)
    # LANGUAGE (11)
    # DATE (39)
    # PERCENT(45)
    # MONEY(41)
    # QUANTITY(48, 49, 40)
    # ORDINAL(42)
    # CARDINAL (others), TIME ()
    MODEL_NAME = 'bert-base-cased'
    MODEL_PATH = 'model/transformers' + "/" + MODEL_NAME
    path = transfer_relative_path(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(path)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # 或者您使用的其他模型
    inputs = tokenizer(question, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    # predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    return predictions.item()


if __name__ == "__main__":
    # simple questions, perform well
    q1 = "Is Rome the capital of Italy?"
    q2 = "Why sky is blue?"
    # questions with option, not good
    q3 = "Is Kosovo a country or a region or others?"
    q4 = "Did USA win the world war II or Germany win?"
    q5 = "Determine which one is correct: A, B, C or D?"
    # questions without Interrogative pronouns, bad
    q6 = "Tell the reason that sky is blue."
    q7 = "distance between earth and moon?"
    # open questions with be, do, can, should...
    q8 = "how are you?"
    q9 = "The method that is able to make a time machine?"
    q10 = "The method to make a time machine?"
    q11 = "can you pass the exam?"
    q12 = "how can we extract answer entities from an answer text?"
    q13 = "when is time machine invented?"
    # others
    q14 = "The capital of Italy is?"
    q15 = "Managua is not the capital of Nicaragua. Yes or no?"
    q16 = "the capital of nicaragua is..."
    q17 = "sky isn't blue, right?"
    q18 = "who is the inventor of smart phone?"
    q19 = "where is Van Gogh from?"
    q20 = "what is most expensive painting"
    q21 = "which language used by most people?"
    q22 = "what lead to the World War I?"
    q23 = "what is GDP of Netherlands in 2022?"
    q24 = "Mozart's nationality?"

    questions = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22,
                 q23, q24]

    for q in questions:
        q_type = classify_question(q)
        print(q)
        if q_type == 0:
            print("open question")
            label = classify_entity_question(q)
            print(label)
        elif q_type == 1:
            print("boolean question")
