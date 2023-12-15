from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
import wikipediaapi

def get_wikipedia_content(page_titles):
    # Use Wikipedia API to get content from multiple pages
    # wikiapi根据关键词查询相关页面内容并进行存储
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
    content = ""

    for title in page_titles:
        page_py = wiki_wiki.page(title)
        content += page_py.text + "\n"

    return content

def generative_summarization(content, keywords, max_length=256, chunk_size=512):
    # Concatenate keywords and content for summarization
    # 对所有收集到的内容进行生成式总结
    input_text = f"Summarize: {', '.join(keywords)}. Content: {content}"

    # Tokenize the input
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn', legacy=False)

    # Load pre-trained BART model for summarization
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Generate summary
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=chunk_size, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


# Function to encode a text using BERT
def encode_text(text):


    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize and encode the text as per the model's requirements
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    # Get the embeddings for the text
    with torch.no_grad():
        outputs = model(**inputs)
    # The last_hidden_state is the embeddings for all tokens
    # Take the mean of the token embeddings to get a single vector for the text
    return outputs.last_hidden_state.mean(dim=1)

# Function to calculate the cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2).item()



QA_llama = "Roma or Romani people, commonly known as Gypsies, are an ethnic group living mostly in Europe and the Americas. Roma is the name of an ancient Egyptian High Priest of Amun and a Brazilian football player. The word Roma is also used to refer to Rome, the capital of Italy."

# Example usage
# 从Task1获得问题和答案的entity 和 他们的link
page_titles = ["Roma", "Italy"]
keywords = ["Italy", "capital", "Roma"]

# Step 1: Get content from multiple Wikipedia pages
content = get_wikipedia_content(page_titles)

# Step 2: Perform generative summarization with keyword attention
wiki_summary = generative_summarization(content, keywords, max_length=256, chunk_size=512)

print(f"Generative Summary:\n{wiki_summary}")

# Encode the texts  summary和llama的输出
wiki_vec = encode_text(wiki_summary)
llama_vec = encode_text(QA_llama)

# Compute the cosine similarity
similarity = cosine_similarity(wiki_vec, llama_vec)

if similarity>0.7:
  print(f"AC!")
else:
  print(f"WA!")


print(f"The similarity score between the texts is: {similarity}")
