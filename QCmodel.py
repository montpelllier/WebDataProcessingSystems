# 0 PERSON (29, 30, 31)

# 1 NORP (Nationalities/religious/political groups) NORP （国籍/宗教/政治团体）(16)

# 2 ORG (Organization) ORG （组织）(28)

# 3 GPE (Countries/cities/states) GPE （国家/城市/州）(32,33,36)

# 4 LOC (Location) LOC （位置）(34, 35)

# 5 FAC, PRODUCT, WORK_OF_ART, LAW(2,3,5,9,10,14,15,22)

# 6 EVENT (8)

# 7 LANGUAGE (11)

# 8 DATE (39)

# 9 PERCENT(45)

# 10 MONEY(41)

# 11 QUANTITY(48, 49, 40)

# 12 ORDINAL(42)

# 13 TIME, CARDINAL (others)(42)
import nltk
from nltk.corpus import nps_chat
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

posts = nps_chat.xml_posts()[:5000]


def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


# identify yes/no questions
def is_yes_no_question(question):
    # 首先使用nltk的分类器
    featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # print(nltk.classify.accuracy(classifier, test_set))
    return classifier.classify(dialogue_act_features(question))

#如果是yes/no问题，直接返回结果, 否则使用bert模型
def Bert(text):
    MODEL_NAME = 'bert-base-cased'
    MODEL_PATH = 'model/transformers' + "/" + MODEL_NAME

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # 或者您使用的其他模型

    # 准备输入数据
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # 生成预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    entity_label = entity_labels_transfer(predictions.item())

    return ("entity labels:", entity_label)

def entity_labels_transfer(label):
    if label == 0:
        return "PERSON"
    elif label == 1:
        return "NORP"
    elif label == 2:
        return "ORG"
    elif label== 3:
        return "GPE"
    elif label== 4:
        return "LOC"
    elif label== 5:
        return "FAC, PRODUCT, WORK_OF_ART, LAW"
    elif label== 6:
        return "EVENT"
    elif label== 7:
        return "LANGUAGE"
    elif label== 8:
        return "DATE"
    elif label== 9:
        return "PERCENT"
    elif label== 10:
        return "MONEY"
    elif label== 11:
        return "QUANTITY"
    elif label== 12:
        return "ORDINAL"
    elif label== 13:
        return "TIME, CARDINAL (others)"


def question_classification(text):
    result = is_yes_no_question(text)
    if result == 'ynQuestion':
        # print(result)
        return result
    else:
        result = Bert(text)
        # print(result)
        return result
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

    questions = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19,q20,q21,q22,q23,q24]

# test
for q in questions:
    # 生成预测
    result = question_classification(q)
    print(q, result)


# train_set = []
# for post in posts:
#     if post.get('class') == 'whQuestion' or post.get('class') == 'ynQuestion':
#        train_set.append((post.text, post.get('class')))


# print(train_set)
