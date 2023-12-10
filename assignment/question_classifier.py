# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

text1 = "Is Rome the capital of Italy?"
text2 = "why sky is blue?"


def classify_question(text):
	# 0: open question; 1: boolean question
	tokenizer = AutoTokenizer.from_pretrained("PrimeQA/tydiqa-boolean-question-classifier")
	model = AutoModelForSequenceClassification.from_pretrained("PrimeQA/tydiqa-boolean-question-classifier")

	inputs = tokenizer(text, return_tensors="pt")
	outputs = model(**inputs)
	question_type = outputs.logits.argmax(dim=-1).item()

	return question_type


if __name__ == "__main__":
	q_type = classify_question(text1)
	if q_type == 0:
		print("open question")
	elif q_type == 1:
		print("boolean question")

	q_type = classify_question(text2)
	if q_type == 0:
		print("open question")
	elif q_type == 1:
		print("boolean question")
