import stanza

from assignment.question_classifier import *
from assignment.sentence_similarity_calculator import *

# test
questions = "Is Rome the capital of Italy?"

answers = ("surely it is. but many don’t know this fact that Italy was not always called as Italy. Before Italy came "
           "into being in 1861, it had several names including Italian Kingdom, Roman Empire and the Republic of "
           "Italy among others. If we start the chronicle back in time, then Rome was the first name to which Romans "
           "were giving credit. Later this city became known as “Caput Mundi” or the capital of the world...")

# stanza.download('en')  # download English model
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,mwt,pos,lemma', download_method=None)
# initialize English neural pipeline
positive_words = ["yes", "certain", "sure", "indeed", "affirm", "agree", "positive", "correct", "right", "definite",
                  "surely"]
negative_words = ["no", "not", "never", "none", "neither", "nor", "without", "deny", "refuse", "reject", "incorrect",
                  "wrong"]


def recognite(doc):
	# doc = nlp(text)

	for sentence in doc.sentences:
		for entity in sentence.ents:
			print(f"entity: {entity.text}, type: {entity.type}")


ans_doc = nlp(answers)
ques_doc = nlp(questions)

# print("question:")
# recognite(ans_doc)
# print("answer:")
# recognite(ques_doc)

if classify_question(questions) == 0:
	# todo: open question
	print("open question")
else:
	# todo: boolean question. using keyword?
	print("boolean question")
	# ans_sentences = [sentence.text for sentence in ans_doc.sentences]
	pair_list = [(questions, answer_sentence.text) for answer_sentence in ans_doc.sentences]
	similarities = cal_sentence_similarity(pair_list)
	# print(list(ans_doc.sentences))
	ans_score = 0
	for i, sentence in enumerate(ans_doc.sentences):
		# check sentences highly related with question or short enough
		if similarities[i] >= 0.4 or len(sentence.words) <= 6:
			print("match sentence:", sentence.text)
			sentence_score = 0
			for word in sentence.words:
				print(i, word.text, word.lemma)
				if word.lemma in positive_words:
					sentence_score += 1
				elif word.lemma in negative_words:
					sentence_score -= 1
			if sentence_score > 0:
				ans_score += 1
			elif sentence < 0:
				ans_score -= 1

	final_ans = "NOT GIVEN/NO IDEA"
	if ans_score > 0:
		final_ans = "YES"
	elif ans_score < 0:
		final_ans = "NO"

	print("ANSWER: ", final_ans)
