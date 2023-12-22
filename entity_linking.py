import re

import stanza
from Levenshtein import distance as levenshtein
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

from assignment.answer_extractor import get_entities
from assignment.html_parser import *

stanza.download('en')  # download English model
# initialize English neural pipeline
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', download_method=None)
URL = "https://www.wikidata.org/w/api.php"
NIL = {'id': None, 'name': None, 'link': None}


# 对于每个entity mention，生成一组候选entity

def generate_entity_candidate(entity, num=10):
    print("mention:", entity)

    S = requests.Session()

    PARAMS = {
        "action": "wbsearchentities",
        "search": entity,
        "limit": num,
        "language": "en",
        "format": "json",
        "type": "item"
    }

    R = S.get(url=URL, params=PARAMS)
    candidate_list = R.json()['search']  # {"searchinfo"=entity,"search"=list(dict)}
    candidate_final_list = []
    for candidate in candidate_list[:]:
        wikipedia_link = get_wikipedia_link(candidate)
        # 如果没有英文链接，将candidate从列表删除
        if wikipedia_link:
            candidate_final_list.append(
                {'id': candidate['id'], 'name': candidate['display']['label']['value'], 'link': wikipedia_link})
        else:
            candidate_list.remove(candidate)
    # print(candidate_final_list)
    return candidate_final_list


def get_wikipedia_link(candidate):
    S = requests.Session()
    wikidata_id = candidate['id']
    PARAMS = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "languages": "en",
        "format": "json"
    }
    R = S.get(url=URL, params=PARAMS)
    data = R.json()
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


# 查询entity在Wikipedia被链接的次数,时间太长了
# def get_entity_popularity(wikipedia_link):
#     session = requests.Session()
#     PARAMETERS = {
#         "action": "query",
#         "format": "json",
#         "list": "backlinks",
#         "bltitle": wikipedia_link,
#         "bllimit": "max"
#     }

#     response = session.get(url=URL, params=PARAMETERS)
#     data = response.json()

#     backlinks = data['query']['backlinks']  # Find all pages that link to the given page.

#     # #解析http页面链接
#     # response = requests.get(wikipedia_link)
#     # soup = BeautifulSoup(response.text, 'html.parser')
#     # backlinks = soup.select('a[class="mw-redirect"]')
#     link_count = len(backlinks)
#     print(link_count)
#     # 如果有更多的反向链接，继续查询
#     while 'continue' in data:  # 但耗时
#         PARAMETERS['blcontinue'] = data['continue']['blcontinue']
#         response = session.get(url=URL, params=PARAMETERS)
#         data = response.json()
#         backlinks.extend(data['query']['backlinks'])
#         link_count += len(data['query']['backlinks'])
#         print(link_count)
#     return link_count


def candidates_ranking(candidates_list, mention, context):
    score = []
    for candidate in candidates_list:
        # wikipedia_link = candidate['link']
        name = candidate['name']

        content = get_candidate_context(candidate)
        # print(content)
        candidate_context = extract_content_with_entity(content, mention)
        similarity = compute_similarity_bow(candidate_context, context)
        string_match_score = levenshtein_distance(mention, name)

        overall = 0.6 * similarity + 0.4 * string_match_score
        score.append(overall)
        # print("score:", overall)
    # 排序
    max_value = max(score)
    max_index = score.index(max_value)

    return candidates_list[max_index]


def extract_content_with_entity(content, entity, max_n=1):
    """
    extract sentences containing keywords
    """
    # print(content)
    sentences = sent_tokenize(content)
    # print(sentences)
    selected_sentences = ""
    n = 0
    for sent in sentences:
        # print(sent)
        if str.lower(entity) in str.lower(sent):
            n += 1
            selected_sentences += sent + " "
            if n == max_n:
                break

    # Remove trailing space
    selected_sentences = selected_sentences.strip()
    # print(selected_sentences)
    return selected_sentences


# 从Wikipedia获取candidate entity上下文
def get_candidate_context(candidate):
    url = candidate['link']
    soup = parse_url(url)
    whole_content = get_wikipedia_page_content(soup)
    whole_content = ' '.join(whole_content)
    selected_content = extract_content_with_entity(whole_content, candidate['name'], max_n=3)
    return selected_content


# 从LLM获取mention上下文
def get_mention_context(sentences, mention):
    contexts = set()
    for sent in sentences:
        for ent in sent.ents:
            if str.lower(ent.text) == str.lower(mention):
                contexts.add(sent.text)
    return " ".join(list(contexts))


def compute_similarity_bow(context_a, context_b):
    # 词袋模型、余弦相似度
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([context_a, context_b])
    similarity = cosine_similarity(vectors[0], vectors[1])

    return similarity


def levenshtein_distance(mention, candidate):
    # 去除非字母字符
    mention = re.sub(r'[^a-zA-Z0-9]', '', mention)
    candidate = re.sub(r'[^a-zA-Z0-9]', '', candidate)
    # 计算Levenshtein距离
    dist = levenshtein(mention.lower(), candidate.lower())
    max_len = max(len(mention), len(candidate))
    return 1 - (dist / max_len)


def link_entity(sentences, entity):
    candidates_list = generate_entity_candidate(entity)
    if candidates_list:
        context = get_mention_context(sentences, entity)
        select = candidates_ranking(candidates_list, entity, context)
    # print(candidates_list)
    else:
        select = NIL
    return select


# if __name__ == '__main__':
#     q = "Managua is not the capital of Nicaragua. Yes or no?"
#     a = ("Most people think Managua is the capital of Nicaragua. However, Managua is not the capital of Nicaragua. The "
#          "capital of Nicaragua is Managua. The capital of Nicaragua is Managua. Managua is the capital of Nicaragua. "
#          "The capital")

#     q_doc = nlp(q)
#     a_doc = nlp(a)
#     sentences = q_doc.sentences + a_doc.sentences

#     ents = set(get_entities(q_doc) + get_entities(a_doc))
#     for ent in ents:
#         print(ent)
#         link_entity(sentences, ent)

# candidates_list = generate_entity_candidate("nicaragua")
# context = get_mention_context(sentences, "nicaragua")
# select = candidates_ranking(candidates_list, "nicaragua", context)


def entity_linking(question, answer):
    q = question
    a = answer

    q_doc = nlp(q)
    a_doc = nlp(a)
    sentences = q_doc.sentences + a_doc.sentences

    ents = set(get_entities(q_doc) + get_entities(a_doc))
    # print("mention:", ents)

    entity_map = {}
    for ent in ents:
        if ent.text not in entity_map.keys():
            entity_map[ent.text] = link_entity(sentences, ent)
    # print("entity_map:", entity_map)
    return entity_map


def question_entity_linking(question):
    q_doc = question
    ents = set(get_entities(q_doc))
    q_link = []
    for ent in ents:
        link = link_entity(q_doc.sentences, ent)
        q_link.append(link['link'])
    return q_link

    # return link_entity(q_doc.sentences, ent)
# question_entity_linking("Managua is not the capital of Nicaragua. Yes or no?")


