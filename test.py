# 1. preprocessing
# 1) tokenization
# 2) normalization (stemming 词干提取 / lemmatization 词性还原)
# 3) stop words removal & 标点符号
# 4) pos tagging 词性标注

import nltk
from nltk.corpus import stopwords, wordnet
import re
from nltk.stem import WordNetLemmatizer
from stanfordcorenlp import StanfordCoreNLP

text = """one of the most visited cities in Europe. everyone wants to see Rome for its unique attractions, historical buildings and artistic masterpieces. The city has a rich history that dates back to ancient times. It was ruled by famous dynasties and is famous for many historical monuments such as the Colosseum, the Pantheon, the Trevi Fountain and more. The best thing about Rome is that it offers a lot of interesting activities at a low price so everyone can afford to visit this city./
Here are some things you can do in Rome: Visit the Vatican Museums. The Vatican City is home to one of the most famous museums in the world, the Vatican Museums. The museum houses many artistic masterpieces and historical artifacts that date back to ancient times. You will find here a large collection of sculptures, paintings, tapestries and more.
The Sistine Chapel is one of the most important parts of the museum, as it contains Michelangelo’s famous frescoes on its ceiling. The museum also houses other artistic masterpieces such as Raphael’s “Madonna di Foligno” or Bern"""


words = nltk.word_tokenize(text)
filtered_words = [word for word in words if word not in stopwords.words('english')]
filtered_words = re.sub(r'[^a-zA-Z0-9\s]', '', str(filtered_words)) #如part-of-speech这种tokenize是会识别为整体的，如果先去除标点就拆掉了
filtered_words = nltk.word_tokenize(filtered_words)
lemma_words = []
lemmatizer = WordNetLemmatizer()
for word, pos in nltk.pos_tag(filtered_words):
    if pos.startswith('J'):
        wordnet_pos = wordnet.ADJ
    elif  pos.startswith('V'):
        wordnet_pos = wordnet.VERB
    elif  pos.startswith('N'):
        wordnet_pos = wordnet.NOUN
    elif  pos.startswith('R'):
        wordnet_pos = wordnet.ADV
    else:
        wordnet_pos = wordnet.NOUN #?
    lemma_words.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

print(lemma_words)
