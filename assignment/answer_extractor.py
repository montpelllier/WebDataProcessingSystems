import stanza
from question_classifier import *

# 示例数据
questions = "Is Rome the capital of Italy?"

answers = ("surely it is but many don’t know this fact that Italy was not always called as Italy. Before Italy came "
           "into being in 1861, it had several names including Italian Kingdom, Roman Empire and the Republic of "
           "Italy among others. If we start the chronicle back in time, then Rome was the first name to which Romans "
           "were giving credit. Later this city became known as “Caput Mundi” or the capital of the world...")

# stanza.download('en')  # download English model
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,ner')  # initialize English neural pipeline


def recognite(text):
	# 处理文本
	doc = nlp(text)
	# 提取命名实体
	for sentence in doc.sentences:
		print(sentence)
		for entity in sentence.ents:
			print(f"entity: {entity.text}, type: {entity.type}")


print("question:")
recognite(questions)
print("answer:")
recognite(answers)

if classify_question(questions) == 0:
	# todo: open question
	pass
else:
	# todo: boolean question
	pass

