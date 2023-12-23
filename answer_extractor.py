import copy

from question_classifier import *
from sentence_similarity_calculator import *

type_list = ["PERSON", "NORP", "ORG", "GPE", "LOC", ["FAC", "PRODUCT", "WORK_OF_ART", "LAW"], "EVENT", "LANGUAGE",
             "DATE", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", ["CARDINAL", "TIME"]]

positive_words = ["yes", "certain", "sure", "indeed", "affirm", "agree", "positive", "correct", "right", "definite",
                  "surely", "certainly", "definitely"]
negative_words = ["no", "not", "never", "none", "neither", "nor", "without", "deny", "refuse", "reject", "incorrect",
                  "wrong"]
# 选取名词、代词、副词、形容词、动词作为keyword
key_pos = ["NOUN", "PROPN", "ADJ", "ADV", "VERB"]


def get_entities(doc):
    entities = []
    for sentence in doc.sentences:
        for entity in sentence.ents:
            entities.append(entity.text)
    return entities


def extract_keywords_by_pos(doc):
    keywords = set()
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in key_pos:
                keywords.add(word.lemma)
    return keywords


def extract_keywords(doc):
    # merge two sets as keywords set
    key_entities = set(get_entities(doc))
    key_words = extract_keywords_by_pos(doc)
    merged_set = copy.deepcopy(key_entities)

    for word in key_words:
        if not any(str.lower(word) in str.lower(entity).split(" ") for entity in key_entities):
            merged_set.add(word)

    return merged_set


def cal_distance_score(sent, keywords, ent):
    word_center = (ent.end_char - ent.start_char) / 2

    keywords = [word.lower() for word in keywords]
    keyword_center_list = {}
    for token in sent.tokens:
        if token.text in keywords:
            center = (token.end_char - token.start_char) / 2
            keyword_center_list[token.text] = center

    distance = 0
    for keyword in keywords:
        if keyword in keyword_center_list.keys():
            keyword_center = keyword_center_list[keyword]
            distance += abs(word_center - keyword_center) / len(sent.text) * 1.5
        else:
            distance += 3

    score = 3 - distance / len(keywords)
    return score


def get_type_score(entity_type, question_type):
    if isinstance(question_type, list):
        return 3 if entity_type in question_type else 0
    return 3 if entity_type == question_type else 0


def get_ans_entity_candidates(ans_doc, q_ent_type, keywords, entity_linking):
    candidates = []
    keywords = {keyword.lower() for keyword in keywords}
    for sentence in ans_doc.sentences:
        for entity in sentence.ents:
            entity_text = str.lower(entity.text)
            if entity.text not in entity_linking.keys():
                continue
            # assume answer entity won't appear in the question.
            if entity_text not in keywords:
                type_score = get_type_score(entity.type, q_ent_type)
                distance_score = cal_distance_score(sentence, keywords, entity)
                candidates.append((entity.text, type_score + distance_score))

    return candidates


def extract_boolean_answer(question, ans_doc):
    pairs = [(question, answer_sentence.text) for answer_sentence in ans_doc.sentences]
    similarities = cal_sentence_similarity(pairs)
    ans_score = 0

    for i, sentence in enumerate(ans_doc.sentences):
        # check sentences highly related with question or short enough
        if similarities[i] >= 0.75 or len(sentence.words) <= 6:
            # print("match sentence:", sentence.text, sentence.sentiment)
            sentence_score = sentence.sentiment - 1  # 0,1,2 negative, neutral, positive

            if sentence_score == 0:
                for word in sentence.words:
                    word_score = 0
                    if word.lemma in positive_words:
                        word_score = 1
                    elif word.lemma in negative_words:
                        word_score = -1

                    if sentence_score == 0 and word_score != 0:
                        sentence_score += word_score
                    elif word_score != 0:
                        sentence_score *= word_score
            ans_score += sentence_score * similarities[i]

    # assume positive if no obvious negative expression
    if ans_score >= 0:
        final_ans = "yes"
    else:
        final_ans = "no"

    return final_ans


def get_ent_type(question):
    ent_type = classify_entity_question(question)
    return type_list[ent_type]


def extract_entity_answer(ques_doc, ans_doc, ent_type, entity_linking):
    # use word type, similarity to question, distance to keywords

    # get keywords from question
    keywords = extract_keywords(ques_doc)
    # print("keywords:", keywords)
    # select candidate entities with type score and distance score.
    candidates = get_ans_entity_candidates(ans_doc, ent_type, keywords, entity_linking)
    if not candidates:
        return None

    pairs = [(ques_doc.text, candidate[0]) for candidate in candidates]
    similarities = cal_sentence_similarity(pairs)
    # add similarity
    candidates = [(tup[0], tup[1] + val) for tup, val in zip(candidates, similarities)]
    # print(candidate)

    entity = max(candidates, key=lambda x: x[1])
    return entity[0]


def extract_answer(ques_doc, ans_doc, entity_linking):
    question = ques_doc.text
    question_type = classify_question(question)
    if question_type == 0:
        # open question. select from entity candidates
        print("type: entity question")
        return extract_entity_answer(ques_doc, ans_doc, get_ent_type(question), entity_linking)
    else:
        # boolean question. use keyword
        print("type: boolean question")
        return extract_boolean_answer(question, ans_doc)
