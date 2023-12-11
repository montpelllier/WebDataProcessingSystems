import stanza

from assignment.question_classifier import *
from assignment.sentence_similarity_calculator import *


def recognite(doc):
	# doc = nlp(text)

	for sentence in doc.sentences:
		for entity in sentence.ents:
			print(f"entity: {entity.text}, type: {entity.type}")


def extract_answer(question, ans_doc):
	# print(q_doc)
	if classify_question(question) == 0:
		# open question. select from entity candidates
		print("open question")
		for sentence in ans_doc.sentences:
			for entity in sentence.ents:
				print(f"entity: {entity.text}, type: {entity.type}")

	# check type, similarity...
	else:
		# boolean question. use keyword
		print("boolean question")
		# ans_sentences = [sentence.text for sentence in ans_doc.sentences]
		pair_list = [(question, answer_sentence.text) for answer_sentence in ans_doc.sentences]
		similarities = cal_sentence_similarity(pair_list)
		# print(list(ans_doc.sentences))
		ans_score = 0
		for i, sentence in enumerate(ans_doc.sentences):
			# check sentences highly related with question or short enough
			if similarities[i] >= 0.75 or len(sentence.words) <= 6:
				print("match sentence:", sentence.text, sentence.sentiment)
				sentence_score = sentence.sentiment - 1  # 0,1,2 negative, neutral, positive

				if sentence_score == 0:
					for word in sentence.words:
						word_score = 0
						if word.lemma in positive_words:
							word_score = 1
						elif word.lemma in negative_words:
							word_score = -1

						if sentence_score == 0 and word_score != 0:
							sentence_score += word_score
						elif word_score != 0:
							sentence_score *= word_score
				ans_score += sentence_score * similarities[i]
				print(ans_score)
				# if sentence_score > 0:  # positive
				# 	ans_score += similarities[i]
				# elif sentence_score < 0:  # negative
				# 	ans_score -= similarities[i]

		final_ans = "not given"
		if ans_score > 0:
			final_ans = "yes"
		elif ans_score < 0:
			final_ans = "no"

		return final_ans


# stanza.download('en')  # download English model
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,mwt,pos,lemma,sentiment', download_method=None)
# initialize English neural pipeline
positive_words = ["yes", "certain", "sure", "indeed", "affirm", "agree", "positive", "correct", "right", "definite",
                  "surely"]
negative_words = ["no", "not", "never", "none", "neither", "nor", "without", "deny", "refuse", "reject", "incorrect",
                  "wrong"]

# print("question:")
# recognite(ans_doc)
# print("answer:")
# recognite(ques_doc)


if __name__ == "__main__":
	# test
	q = "Is Rome the capital of Italy?"
	a = (
		"surely it is. but many don’t know this fact that Italy was not always called as Italy. Before Italy came "
		"into being in 1861, it had several names including Italian Kingdom, Roman Empire and the Republic of "
		"Italy among others. If we start the chronicle back in time, then Rome was the first name to which Romans "
		"were giving credit. Later this city became known as “Caput Mundi” or the capital of the world...")
	a_doc = nlp(a)
	q_doc = nlp(q)
	print(extract_answer(q, a_doc))

	q = "Managua is not the capital of Nicaragua. Yes or no?"
	a = ("Most people think Managua is the capital of Nicaragua. However, Managua is not the capital of Nicaragua. The "
	     "capital of Nicaragua is Managua. The capital of Nicaragua is Managua. Managua is the capital of Nicaragua. "
	     "The capital")
	a_doc = nlp(a)
	q_doc = nlp(q)
	print(extract_answer(q, a_doc))

	q = "sky isn't blue, right?"
	a = ("The statement \"the sky isn't blue\" is not accurate. The Earth's atmosphere, particularly the gases and "
	     "particles in the air, scatters sunlight, making the sky appear blue. This phenomenon is known as Rayleigh "
	     "scattering, named after Lord Rayleigh, who first described it in the late 19th century. The blue color we "
	     "see in the sky is a result of the scattering of sunlight by the tiny molecules of gases in the atmosphere, "
	     "such as nitrogen and oxygen. The shorter, blue wavelengths are scattered in all directions, while the "
	     "longer, red wavelengths pass straight through the atmosphere with little scattering, which is why the sky "
	     "typically appears blue during the daytime. It's worth noting that the color of the sky can change depending "
	     "on the time of day and atmospheric conditions. For example, during sunrise and sunset, the sky can take on "
	     "hues of red, orange, and pink, due to the scattering of light by atmospheric particles. However, "
	     "the blue color of the sky remains a constant feature of the Earth's atmosphere under normal conditions.")
	a_doc = nlp(a)
	q_doc = nlp(q)
	print(extract_answer(q, a_doc))
