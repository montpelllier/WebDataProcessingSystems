import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords, wordnet
import re
import numpy as np
from Levenshtein import distance as levenshtein

# 对于每个entity mention，生成一组候选entity
def generate_entity_candidate(entity, num=50):
    S = requests.Session()

    # URL = "https://en.wikipedia.org/w/api.php"
    URL = "https://www.wikidata.org/w/api.php"
    # URL = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=bond&format=json&language=en&uselang=en&type=item&limit=20"

    PARAMS = {
        "action": "wbsearchentities",
        "search": entity,
        "limit": num,
        "language":"en",
        "format": "json",
        "type": "item"
    }

    R = S.get(url=URL, params=PARAMS)
    candidate_list = R.json()['search']  #{"searchinfo"=entity,"search"=list(dict)}
    # wikidata_url_list = [candidate['url'] for candidate in candidate_list]
    # print(candidate_list)
    candidate_final_list  = []
    for candidate in candidate_list[:]:
        wikipedia_link = get_wikipedia_link(candidate)
        #如果没有英文链接，将candidate从列表删除
        if wikipedia_link:
            candidate_final_list.append({'id': candidate['id'], 'name': candidate['display']['label']['value'], 'link': wikipedia_link})
        else:
            candidate_list.remove(candidate)
    return candidate_final_list

def get_wikipedia_link(candidate):
    S = requests.Session()
    # wikidata_url = 'https:' + candidate['url']
    wikidata_id = candidate['id']
    URL = "https://www.wikidata.org/w/api.php"
    # https: // www.wikidata.org / w / api.php?action = wbgetentities & ids = Q42 & format = json
    PARAMS = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "languages": "en",
        "format": "json"
    }
    R = S.get(url=URL, params=PARAMS)
    data = R.json()
    # print(data)
    sitelinks = data['entities'][wikidata_id]['sitelinks']
    en_site = 'enwiki'
    if en_site in sitelinks:
        title = data['entities'][wikidata_id]['sitelinks'][en_site]['title']
        # 构建Wikipedia链接
        wikipedia_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    # 如果不是英文链接
    else:
        wikipedia_url = None
    return wikipedia_url
    #
    #
    # wikipedia_url = sitelinks.get('enwiki', {}).get('url')
    # print(sitelinks)
    # print(wikipedia_url)
    # if wikipedia_url:
    #     return wikipedia_url
    # else:#如果没有英语链接，返回其他语言链接：法、西、德、意
    #     for lang in ['frwiki', 'eswiki', 'dewiki', 'itwiki']:
    #         wikipedia_url = sitelinks.get(lang, {}).get('url')
    #         if wikipedia_url:
    #             return wikipedia_url
    #     #如果没有法、西、德、意链接，默认返回第一个
    #     wikipedia_url = next(iter(sitelinks.values()))['url']
    #     return wikipedia_url

# 查询entity在Wikipedia被链接的次数,时间太长了
def get_entity_popularity(wikipedia_link):
    session = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"

    PARAMETERS = {
        "action": "query",
        "format": "json",
        "list": "backlinks",
        "bltitle": wikipedia_link,
        "bllimit": "max"
    }

    response = session.get(url=URL, params=PARAMETERS)
    data = response.json()

    backlinks = data['query']['backlinks'] #Find all pages that link to the given page.

    # #解析http页面链接
    # response = requests.get(wikipedia_link)
    # soup = BeautifulSoup(response.text, 'html.parser')
    # backlinks = soup.select('a[class="mw-redirect"]')
    link_count = len(backlinks)
    print(link_count)
    # 如果有更多的反向链接，继续查询
    while 'continue' in data: #但耗时
        PARAMETERS['blcontinue'] = data['continue']['blcontinue']
        response = session.get(url=URL, params=PARAMETERS)
        data = response.json()
        backlinks.extend(data['query']['backlinks'])
        link_count += len(data['query']['backlinks'])
        print(link_count)
    return link_count

def candidates_ranking(candidates_list, mention):
    # context_score = rank_by_similarity_bow(candidates_list,mention)
    # string_match_score = levenshtein_distance(mention)
    # print(score)
    mention = "Rome"
    score = []
    for candidate in candidates_list:
        wikipedia_link = candidate['link']
        name = candidate['name']
        content = get_wikipedia_page_content(wikipedia_link)
        candidate_context = extract_content_with_entity(content, mention)
        print(candidate_context)
        #待与NER合并
        mention_context = """one of the most visited cities in Europe. everyone wants to see Rome for its unique attractions, historical buildings and artistic masterpieces.
    The city has a rich history that dates back to ancient times. It was ruled by famous dynasties and is famous for many historical monuments such as the Colosseum, the Pantheon, the Trevi Fountain and more. The best thing about Rome is that it offers a lot of interesting activities at a low price so everyone can afford to visit this city.
    Here are some things you can do in Rome:
    Visit the Vatican Museums
    The Vatican City is home to one of the most famous museums in the world, the Vatican Museums. The museum houses many artistic masterpieces and historical artifacts that date back to ancient times. You will find here a large collection of sculptures, paintings, tapestries and more.
    The Sistine Chapel is one of the most important parts of the museum, as it contains Michelangelo’s famous frescoes on its ceiling. The museum also houses other artistic masterpieces such as Raphael’s “Madonna di Foligno” or Bern"""
        similarity = compute_similarity_bow(candidate_context, mention_context)
        string_match_score = levenshtein_distance(mention,name)
        overall = 0.6*similarity+0.4*string_match_score
        score.append(overall)
        print(overall)
    #排序
    max_value = max(score)
    max_index = score.index(max_value)

    return candidates_list[max_index]

def rank_by_similarity_bow(candidates_list, mention):
    mention = "Rome"
    score = []
    for candidate in candidates_list:
        wikipedia_link = candidate['link']
        name = candidate['name']
        content = get_wikipedia_page_content(wikipedia_link)
        candidate_context = extract_content_with_entity(content, mention)
        print(candidate_context)
        mention_context = """one of the most visited cities in Europe. everyone wants to see Rome for its unique attractions, historical buildings and artistic masterpieces.
The city has a rich history that dates back to ancient times. It was ruled by famous dynasties and is famous for many historical monuments such as the Colosseum, the Pantheon, the Trevi Fountain and more. The best thing about Rome is that it offers a lot of interesting activities at a low price so everyone can afford to visit this city.
Here are some things you can do in Rome:
Visit the Vatican Museums
The Vatican City is home to one of the most famous museums in the world, the Vatican Museums. The museum houses many artistic masterpieces and historical artifacts that date back to ancient times. You will find here a large collection of sculptures, paintings, tapestries and more.
The Sistine Chapel is one of the most important parts of the museum, as it contains Michelangelo’s famous frescoes on its ceiling. The museum also houses other artistic masterpieces such as Raphael’s “Madonna di Foligno” or Bern"""
        similarity = compute_similarity_bow(candidate_context,mention_context)
        score.append(similarity)
    return score

def get_wikipedia_page_content(url):
    # 发送 GET 请求获取页面内容
    response = requests.get(url)
    # 检查请求是否成功
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # 找到页面内容的标签，这取决于维基百科页面的结构
        # 以下是以类名为 "mw-parser-output" 的 <div> 标签为例
        content_div = soup.find('div', {'class': 'mw-body-content'})
        # 找到该 div 元素下的所有 <p> 元素
        paragraphs = content_div.find_all('p')
        # 提取每个 <p> 元素的文本内容
        paragraph_texts = ""

        for paragraph in paragraphs:
          paragraph_texts += paragraph.get_text() + ' '
        return paragraph_texts
    else:
        print(f"failed request: {response.status_code}")
        return None

def extract_content_with_entity(content, entity ,max_n=1):
    '''
    extract sentences containing keywords
    '''
    sentences = sent_tokenize(content)
    selected_sentences = ""
    n = 0
    for sentence in sentences:
        if entity in sentence:
            n+=1
            selected_sentences += sentence + " "
            if n==max_n:
                break

    # Remove trailing space
    selected_sentences = selected_sentences.strip()
    return selected_sentences

#从Wikipedia获取candiate entity上下文
def get_candidate_context(candidate):
    url = candidate['link']
    whole_content = get_wikipedia_page_content(url)
    selected_content = extract_content_with_entity(whole_content, candidate, max_n=5)
    return selected_content

#从LLM获取mention上下文
def get_mention_context():
    return None

def compute_similarity_bow(context_a, context_b):
    #tdf-if
    # vectorizer = TfidfVectorizer()
    # context_a = context_preprocessing(context_a)
    # context_b = context_preprocessing(context_b)
    # tfidf_matrix = vectorizer.fit_transform([context_a, context_b])
    # similarity = cosine_similarity(tfidf_matrix)

    #词袋模型、余弦相似度
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([context_a, context_b])
    similarity = cosine_similarity(vectors[0], vectors[1])

    return similarity

def context_preprocessing(context):
    words = nltk.word_tokenize(context)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    filtered_words = re.sub(r'[^a-zA-Z0-9\s]', '', str(filtered_words))  # 如part-of-speech这种tokenize是会识别为整体的，如果先去除标点就拆掉了
    filtered_words = nltk.word_tokenize(filtered_words)
    # print(filtered_words)
    # 创建词袋向量化器
    vectorizer = CountVectorizer()

    # 转换为词袋向量
    vector = vectorizer.fit_transform(filtered_words)
    #
    # # 输出词袋向量
    # print(vector.toarray())
    return vector


def compute_similarity_word2vec(mention_context,candidate_context):
    model = KeyedVectors.load_word2vec_format('path/to/GoogleNews-vectors-negative300.bin', binary=True)

    # 定义文本


    # 预处理文本：去除停用词，并分词
    mention_words = context_preprocessing(mention_context)
    candidate_words = context_preprocessing(candidate_context)

    # 计算文本的Word2Vec向量（通过取所有词向量的平均）
    def get_vector(words):
        # 只考虑模型已知的词
        valid_words = [word for word in words if word in model]
        if valid_words:
            return np.mean([model[word] for word in valid_words], axis=0)
        else:
            return np.zeros(model.vector_size)

    mention_vec = get_vector(mention_words)
    candidate_vec = get_vector(candidate_words)

    # 计算两个向量之间的余弦相似度
    cosine_similarity = np.dot(mention_vec, candidate_vec) / (
                np.linalg.norm(mention_vec) * np.linalg.norm(candidate_vec))

    print("Cosine Similarity: ", cosine_similarity)

def levenshtein_distance(mention, candidate):
    print(len(mention))
    # 去除非字母字符
    mention = re.sub(r'[^a-zA-Z]', '', mention)
    candidate = re.sub(r'[^a-zA-Z]', '', candidate)
    # 计算Levenshtein距离
    dist = levenshtein(mention.lower(), candidate.lower())
    max_len = max(len(mention), len(candidate))
    return 1 - (dist / max_len)


if __name__ == '__main__':
    candidates_list = generate_entity_candidate("Rome")
    select = candidates_ranking(candidates_list,"Rome")
    print(select)

