from QCmodel import question_classification
from fact_checking import fact_checking
from example_using_llm import get_completion
def main():

    question = input("Type your question and type ENTER to finish:\n")
    answer = get_completion(question)
    # entity_linking
    questionclassify = question_classification(question)
    # factchecking
    # factcheck = fact_checking(question, entity_question, entity_question_link, extracted_answer)
    print("answer", answer)
    print("question classify:", questionclassify)
    # print("factchecking",factcheck)

if __name__ == "__main__":
    main()