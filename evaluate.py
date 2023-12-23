from sklearn.metrics import accuracy_score, f1_score

from QCmodel import question_classification
from fact_checking import fact_checking
from example_using_llm import get_completion
from answer_extractor import *
from entity_linking import entity_linking, question_entity_linking
import stanza
import nltk

nltk.download('punkt')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,mwt,pos,lemma,sentiment', download_method=None)

# modify the input file path
INPUT_FILE = 'testdata_input.txt'
GT_FILE = 'testdata_gt.txt'
OUTPUT_FILE = 'output.txt'


def main():
    # stanza.download('en')
    # generate question and answer
    # question = input("Type your question and type ENTER to finish:\n")
    # question = "What is the capital of China?"
    with open(INPUT_FILE, 'r') as file:
        questions = file.readlines()
    gt_data = read_output_file(GT_FILE)
    with open(OUTPUT_FILE, 'w') as output_file:
        for q in questions:
            try:
                q = q.strip()
                question_id, question = q.split('\t')
                # print("Question ID:", question_id)
                # print("Question Text:", question)
                # answer = get_completion(question)
                llm_answer = gt_data[question_id]['R']
                answer = llm_answer
                # print("question", question)
                # print("Text returned by the language model", question_id, answer)
                output_file.write(f"{question_id}\tR\"{answer}\"\n")

                # entity_linking
                entity_linking_result = entity_linking(question, answer)
                # questionclassify = question_classification(question)

                # print("Entities extracted", question_id, entity_linking_result)
                # factchecking
                q_doc = nlp(question)
                a_doc = nlp(answer)
                entity_question = get_entities(q_doc)
                entity_question_link = question_entity_linking(q_doc)
                extracted_answer = extract_answer(q_doc, a_doc)
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

    # 读取标准输出和预测输出
    pred_data = read_output_file(OUTPUT_FILE)  # 预测数据

    # 计算指标
    accuracy, f1 = calculate_metrics_forC(gt_data, pred_data)
    print(f"C:Accuracy: {accuracy}")
    print(f"C:F1 Score: {f1}")
    precision, recall, f1 = evaluate_entities_forE(gt_data, pred_data)
    print(f"E:Precision: {precision}")
    print(f"E:Recall: {recall}")
    print(f"E:F1 Score: {f1}")


def read_output_file(filename):
    """读取输出文件并返回字典形式的数据。"""
    data = {}
    with open(filename, 'r') as file:
        content = file.read().strip().split('\n\n')
        for block in content:
            lines = block.strip().split('\n')
            question_id = None
            record = {'R': None, 'A': None, 'C': None, 'E': []}

            for line in lines:
                parts = line.split()
                if not parts:
                    continue  # Skip empty lines

                # 获取问题ID
                if question_id is None:
                    question_id = parts[0]

                # 根据行的开始标记处理数据
                if line.startswith(question_id + " R"):
                    record['R'] = line[len(question_id) + 3:].strip('"')
                elif line.startswith(question_id + " A"):
                    record['A'] = line[len(question_id) + 3:].strip('"')
                elif line.startswith(question_id + " C"):
                    record['C'] = line[len(question_id) + 3:].strip('"')
                elif line.startswith(question_id + " E"):
                    entity, link = line[len(question_id) + 3:].split('\t')
                    record['E'].append((entity.strip('"'), link.strip('"')))

            if question_id:
                data[question_id] = record
    return data


def calculate_metrics_forC(true_data, pred_data):
    """计算准确率和F1分数。"""
    true_labels = []
    pred_labels = []
    for qid in true_data:
        if qid in pred_data:
            true_labels.append(true_data[qid]['C'])
            pred_labels.append(pred_data[qid]['C'])

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, pos_label="correct")

    return accuracy, f1


def evaluate_entities_forE(true_data, pred_data):
    # 初始化计数器
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # 为每个问题计算真实实体和预测实体
    for qid in true_data:
        true_entities = set((e[0], e[1]) for e in true_data[qid]['E'])
        if qid in pred_data and 'E' in pred_data[qid]:
            pred_entities = set((e[0], e[1]) for e in pred_data[qid]['E'])
        else:
            pred_entities = set()

        # 计算真正例、假正例和假负例
        true_positives += len(true_entities & pred_entities)
        false_positives += len(pred_entities - true_entities)
        false_negatives += len(true_entities - pred_entities)

    # 计算精确度、召回率和F1分数
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


if __name__ == "__main__":
    main()
