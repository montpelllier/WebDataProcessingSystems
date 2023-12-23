from fact_checking import fact_checking
from example_using_llm import get_completion
from answer_extractor import get_entities, answer_extractor
from entity_linking import entity_linking, question_entity_linking
import stanza
import nltk

nltk.download('punkt')

# modify the input file path
INPUT_FILE = 'example_input.txt'
OUTPUT_FILE = 'output.txt'


def main():
    # stanza.download('en')
    # generate question and answer
    # question = input("Type your question and type ENTER to finish:\n")
    # question = "What is the capital of China?"
    with open(INPUT_FILE, 'r') as file:
        questions = file.readlines()
    with open(OUTPUT_FILE, 'w') as output_file:
        for q in questions:
            try:
                q = q.strip()
                question_id, question = q.split('\t')
                # print("Question ID:", question_id)
                # print("Question Text:", question)
                answer = get_completion(question)
                # print("question", question)
                # print("Text returned by the language model", question_id, answer)
                output_file.write(f"{question_id}\tR\"{answer}\"\n")

                # entity_linking
                entity_linking_result = entity_linking(question, answer)
                # questionclassify = question_classification(question)

                # print("Entities extracted", question_id, entity_linking_result)
                # factchecking
                q_doc = trans_to_doc(question)
                entity_question = get_entities(q_doc)
                entity_question_link = question_entity_linking(q_doc)
                extracted_answer = answer_extractor(question, answer)
                output_file.write(f"{question_id}\tA\"{extracted_answer}\"\n")

                # test
                # print("q_doc", q_doc)
                # print("question entity",entity_question)
                # print("question entity link",question_id,entity_question_link)
                if extracted_answer == "yes" or extracted_answer == "no":
                    print("extracted answer", question_id, extracted_answer)
                    factcheck = fact_checking(question, entity_question, entity_question_link, extracted_answer)

                else:
                    ans_link = entity_linking_result[extracted_answer]
                    print("extracted answer", question_id, extracted_answer, ans_link)
                    factcheck = fact_checking(question, entity_question, entity_question_link, ans_link)
                # print("answer", answer)
                # print("question classify:", questionclassify)
                # print("Correctness of the answer: ",question_id,factcheck)
                output_file.write(f"{question_id}\tC\"{factcheck}\"\n")
                for entity in entity_question_link:
                    output_file.write(f"{question_id}\tE\"{entity['name']}\"\t\"{entity['link']}\"\n")
            except Exception as e:
                print(question_id, f"An error occurred: {e}. Skipping this question.")


def trans_to_doc(ques):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,mwt,pos,lemma,sentiment', download_method=None)
    q_doc = nlp(ques)
    return q_doc


if __name__ == "__main__":
    main()
