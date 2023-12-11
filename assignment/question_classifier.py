# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def classify_question(text):
	# 0: open question; 1: boolean question
	tokenizer = AutoTokenizer.from_pretrained("PrimeQA/tydiqa-boolean-question-classifier")
	model = AutoModelForSequenceClassification.from_pretrained("PrimeQA/tydiqa-boolean-question-classifier")

	inputs = tokenizer(text, return_tensors="pt")
	outputs = model(**inputs)
	question_type = outputs.logits.argmax(dim=-1).item()

	return question_type


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
	questions = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17]

	for q in questions:
		q_type = classify_question(q)
		print(q)
		if q_type == 0:
			print("open question")
		elif q_type == 1:
			print("boolean question")
