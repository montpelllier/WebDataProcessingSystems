# PERSON (29, 30, 31)

# NORP (Nationalities/religious/political groups) NORP （国籍/宗教/政治团体）(16)

# ORG (Organization) ORG （组织）(28)

# GPE (Countries/cities/states) GPE （国家/城市/州）(32,33,36)

# LOC (Location) LOC （位置）(34, 35)

# FAC, PRODUCT, WORK_OF_ART, LAW(2,3,5,9,10,14,15,22)

# EVENT (8)

# LANGUAGE (11)

# DATE (39)

# TIME ()

# PERCENT(45)

# MONEY(41)

# QUANTITY(48, 49, 40)

# ORDINAL(42)

# CARDINAL (others)
import torch
from transformers import BertModel, BertTokenizer, BertConfig, AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = 'bert-base-cased'
MODEL_PATH = 'model/transformers' + "/" + MODEL_NAME

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # 或者您使用的其他模型

# 准备输入数据
text = "When is Chrismas day" # modify here
inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# 生成预测
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

print("predict labels:", predictions.item())