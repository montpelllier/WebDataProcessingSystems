import pandas as pd
import datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np



DATA_PATH = 'train.jsonl'

# 加载数据集
def load_data(DATA_PATH):
    bq_data = pd.read_json(DATA_PATH, lines=True)
    bq_data = bq_data.sample(n=5000, random_state=1)
    # entity_data = load_dataset('trec', split='train')
    entity_data = pd.DataFrame(load_dataset('trec', split='train'))
    entity_data = entity_data.sample(n=5000, random_state=1)

    # 修改数据集
    bq_data = bq_data.drop(['passage', 'answer', 'title'], axis=1)
    bq_data['labels'] = 'boolean'
    entity_data = entity_data.rename(columns={'text': 'question'})
    entity_data = entity_data.drop(['coarse_label', 'fine_label'], axis=1)
    entity_data['labels'] = 'entity'
    datasets = pd.concat([bq_data, entity_data], ignore_index=True)
    return datasets

    # 查看数据集的前几行
    # print(datasets.head(20))
    # print(datasets.shape[0])

# training model
datasets = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(datasets['question'], datasets['labels'], test_size=0.2, random_state=1)

# 将文本数据转换为数值特征向量
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 训练逻辑回归模型
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
train_score = classifier.score(X_train, y_train)
test_score = classifier.score(X_test, y_test)

print(f'Train Score: {train_score}')
print(f'Test Score: {test_score}')
joblib.dump(classifier, 'classifier.pkl') 
joblib.dump(vectorizer, 'vectorizer.pkl')  # 保存 vectorizer
# 测试模型
# print(classifier.score(X_test, y_test))
# q1 = "Is Rome the capital of Italy?"
# q1_transformed = vectorizer.transform([q1])
# prediction = classifier.predict(q1_transformed)

# print(prediction)