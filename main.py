from QCmodel import question_classification
from fact_checking import fact_checking
from example_using_llm import get_completion
from assignment.answer_extractor import get_entities, answer_extractor
from entity_linking import entity_linking,question_entity_linking
import stanza

def main():
# generate question and answer
    # question = input("Type your question and type ENTER to finish:\n")
    # question = "What is the capital of China?"
    with open('example_input.txt', 'r') as file:
        questions = file.readlines()
    for q in questions:
        q = q.strip()
        question_id, question = q.split('\t')
        print("Question ID:", question_id)
        print("Question Text:", question)
        answer = get_completion(question)
        print("question", question)
        print("answer", answer)

    # entity_linking
        entity_linking_result = entity_linking(question, answer)
        # questionclassify = question_classification(question)

    # factchecking
        q_doc = trans_to_doc(question)
        entity_question = get_entities(q_doc)
        entity_question_link = question_entity_linking(q_doc)
        extracted_answer  = answer_extractor(question,answer)

        # test
        # print("q_doc", q_doc)
        # print("question entity",entity_question)
        print("question entity link",entity_question_link)
        print("extracted answer",extracted_answer)

        factcheck = fact_checking(question, entity_question, entity_question_link, extracted_answer)
        # print("answer", answer)
        # print("question classify:", questionclassify)
        print("factchecking",factcheck)

# 



def trans_to_doc(ques):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,mwt,pos,lemma,sentiment', download_method=None)
    q_doc = nlp(ques)
    return q_doc
    
if __name__ == "__main__":
    main()