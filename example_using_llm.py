from ctransformers import AutoModelForCausalLM
repo = "TheBloke/Llama-2-7B-GGUF"
model_file = "llama-2-7b.Q4_K_M.gguf"
llm = AutoModelForCausalLM.from_pretrained(repo, model_file=model_file, model_type="llama")

def get_completion(prompt):
	while(True):
		if prompt == 'quit':
			break
		print("Computing the answer (takes time)...")
		completion = llm(prompt)
		# print("COMPLETION: %s" % completion)
		return completion

