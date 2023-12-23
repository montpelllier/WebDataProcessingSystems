import nltk
import stanza

from answer_extractor import *
from entity_linking import entity_linking, question_entity_linking
from example_using_llm import get_completion
from fact_checking import fact_checking

nltk.download('punkt')

# modify the input file path
INPUT_FILE = 'example_input.txt'
OUTPUT_FILE = 'output.txt'

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,mwt,pos,lemma,sentiment', download_method=None)


def main():
    with open(INPUT_FILE, 'r') as file:
        # questions = file.readlines()
        questions = [line.rstrip() for line in file if line.rstrip()]
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
        for q in questions:
            try:
                q = q.strip()
                question_id, question = q.split('\t')
                answer = get_completion(question)
                print("question: ", question)
                print("answer: ", answer)
                # output_file.write(f"{question_id}\tR\"{answer}\"\n")
                # question_classify = question_classification(question)

                # fact checking
                q_doc = nlp(question)
                a_doc = nlp(answer)
                # entity_linking
                entity_linking_result = entity_linking(q_doc, a_doc)

                entity_question = get_entities(q_doc)
                entity_question_link = question_entity_linking(q_doc)
                extracted_answer = extract_answer(q_doc, a_doc)
                output_file.write(f"{question_id}\tA\"{extracted_answer}\"\n")

                # test
                if extracted_answer == "yes" or extracted_answer == "no":
                    print("extracted answer", question_id, extracted_answer)
                    factcheck = fact_checking(question, entity_question, entity_question_link, extracted_answer)

                else:
                    ans_link = entity_linking_result[extracted_answer]['link']
                    print("extracted answer", question_id, extracted_answer, ans_link)
                    factcheck = fact_checking(question, entity_question, entity_question_link, ans_link)

                output_file.write(f"{question_id}\tC\"{factcheck}\"\n")
                for query in entity_linking_result.keys():
                    name = entity_linking_result[query]['name']
                    link = entity_linking_result[query]['link']
                    print(name, link)
                    output_file.write(f"{question_id}\tE\"{entity_linking_result[query]['name']}\"\t\"{entity_linking_result[query]['link']}\"\n")

                print("factcheck: ", factcheck)
            except Exception as e:
                print(question_id, f"An error occurred: {e}. Skipping this question.")
            print("------------------------------------------------------------------")
    output_file.close()


if __name__ == "__main__":
    main()
