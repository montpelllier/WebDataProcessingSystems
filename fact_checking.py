import requests
import torch
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_wikipedia_page_content(url):
    # 发送 GET 请求获取页面内容
    response = requests.get(url)
    # 检查请求是否成功
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # 找到页面内容的标签，这取决于维基百科页面的结构
        # 以下是以类名为 "mw-parser-output" 的 <div> 标签为例
        content_div = soup.find('div', {'class': 'mw-body-content'})
        # 找到该 div 元素下的所有 <p> 元素
        paragraphs = content_div.find_all('p')
        # 提取每个 <p> 元素的文本内容
        paragraph_texts = ""

        for paragraph in paragraphs:
            paragraph_texts += paragraph.get_text() + ' '
        return paragraph_texts
    else:
        print(f"failed request: {response.status_code}")
        return None


def extract_content_with_keywords(content, keywords):
    '''
    extract sentences containing keywords
    '''
    sentences = sent_tokenize(content)
    selected_sentences = ""

    for sentence in sentences:
        if all(keyword in sentence for keyword in keywords):
            selected_sentences += sentence + " "

    # Remove trailing space
    selected_sentences = selected_sentences.strip()
    return selected_sentences


def get_boolQ_predict(question, content):
    '''
    Get the answer of bool question using model roberta-large-boolq
    '''
    tokenizer = AutoTokenizer.from_pretrained("nfliu/roberta-large_boolq")
    model_boolQ = AutoModelForSequenceClassification.from_pretrained("nfliu/roberta-large_boolq")

    sequence = tokenizer.encode_plus(question, content, return_tensors="pt", max_length=512, truncation=True)[
        'input_ids']
    logits = model_boolQ(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    proba_yes = round(probabilities[1], 2)
    proba_no = round(probabilities[0], 2)

    # print(f"Question: {question}, Yes: {proba_yes}, No: {proba_no}")

    if proba_yes > proba_no:
        return "yes"
    else:
        return "no"


def encode_text(text):
    sim_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    encoded_text = sim_model.encode(text, convert_to_tensor=True)
    return encoded_text


def get_similarity_score(embedding1, embedding2):
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_scores.item()


def result_similarity_score(text1, text2):
    embedding1 = encode_text(text1)
    embedding2 = encode_text(text2)
    similarity_score = get_similarity_score(embedding1, embedding2)
    return similarity_score


def fact_checking(question, entity_question, entity_question_link, extracted_answer):
    keywords = entity_question
    all_keywords_contents = ""
    for entity_link in entity_question_link:
        entity_content = get_wikipedia_page_content(entity_link)
        keywords_contents = extract_content_with_keywords(entity_content, keywords)
        all_keywords_contents += keywords_contents

    if extracted_answer == "yes" or extracted_answer == "no":
        yesno_boolQ = get_boolQ_predict(question, all_keywords_contents)
        if yesno_boolQ == extracted_answer:
            return ("Correct")
        else:
            return ("Incorrect")

    else:
        extracted_content = get_wikipedia_page_content(extracted_answer)
        extracted_keywords_contents = extract_content_with_keywords(extracted_content, keywords)
        similarity = result_similarity_score(extracted_keywords_contents, all_keywords_contents)
        # print(extracted_keywords_contents, all_keywords_contents, similarity)

        if similarity > 0.7:
            return ("Correct")
        else:
            return ("Incorrect")

# if __name__ == '__main__':

#     # question = "Is Beijing the capital of China?"
#     question = "Why is the sky blue?"

#     # entity_question = ['sky', 'capital', 'Beijing']
#     # entity_question_link = ["https://en.wikipedia.org/wiki/China", "https://en.wikipedia.org/wiki/capital", "https://en.wikipedia.org/wiki/Beijing"]

#     entity_question = ['sky', 'blue']
#     entity_question_link = ["https://en.wikipedia.org/wiki/sky", "https://en.wikipedia.org/wiki/blue"]

#     keywords = entity_question

#     # entity_answer = ['Beijing', 'capital', 'China']

#     # extracted_answer = "yes"
#     extracted_answer = "https://en.wikipedia.org/wiki/sky"


#     all_keywords_contents = ""
#     for entity_link in entity_question_link:
#         entity_content = get_wikipedia_page_content(entity_link)
#         keywords_contents = extract_content_with_keywords(entity_content, keywords)
#         all_keywords_contents += keywords_contents

#     # print(all_keywords_contents)

#     if extracted_answer == "yes" or extracted_answer == "no":
#         yesno_boolQ = get_boolQ_predict(question, all_keywords_contents)
#         if yesno_boolQ == extracted_answer:
#             print("Correct")
#         else:
#             print("Incorrect")

#     else:
#         extracted_content = get_wikipedia_page_content(extracted_answer)
#         extracted_keywords_contents = extract_content_with_keywords(extracted_content, keywords)
#         similarity = result_similarity_score(extracted_keywords_contents, all_keywords_contents)
#         # print(extracted_keywords_contents, all_keywords_contents, similarity)

#         if similarity > 0.7:
#             print("Correct")
#         else:
#             print("Incorrect")
