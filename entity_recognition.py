from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import requests


def recognize_entities(text):
    # 设置斯坦福NER分类器和Java路径
    stanford_classifier = './stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'
    stanford_ner_path = './stanford-ner/stanford-ner.jar'

    st = StanfordNERTagger(stanford_classifier, stanford_ner_path)
    # tokenized_text = word_tokenize(text)
    # classified_text = st.tag(tokenized_text)
    classified_text = st.tag(text)

    # Wikipedia API
    WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

    # 存储实体/Wikipedia URL的字典
    entity_urls = {}

    for entity in classified_text:
        if entity[1] in ["PERSON", "ORGANIZATION", "LOCATION"]:
            # 构建API请求
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": entity[0]
            }

            # 发送请求
            response = requests.get(WIKIPEDIA_API_URL, params=params)
            data = response.json()

            # 解析结果
            if data["query"]["search"]:
                page_id = data["query"]["search"][0]["pageid"]

                # 到Wikipedia页面的链接
                wikipedia_url = f"https://en.wikipedia.org/?curid={page_id}"
                entity_urls[entity[0]] = wikipedia_url

    # 测试
    for entity, url in entity_urls.items():
        print(f"{entity}  {url}")

    return entity_urls


# # 测试
# text = "Is Rome the capital of Italy? Surely it is, " \
#        "but many don’t know this fact that Italy was not always " \
#        "called as Italy. Before Italy came into being in 1861 it had " \
#        "several names including Italian Kingdom, Roman Empire, and the " \
#        "Republic of Italy among others."
#
# recognize_entities(text)

# text
text = ['one', 'visited', 'city', 'Europe', 'everyone', 'want', 'see', 'Rome', 'unique', 'attraction', 'historical', 'building', 'artistic', 'masterpiece', 'The', 'city', 'rich', 'history', 'date', 'back', 'ancient', 'time', 'It', 'rule', 'famous', 'dynasty', 'famous', 'many', 'historical', 'monument', 'Colosseum', 'Pantheon', 'Trevi', 'Fountain', 'The', 'best', 'thing', 'Rome', 'offer', 'lot', 'interest', 'activity', 'low', 'price', 'everyone', 'afford', 'visit', 'city', 'Here', 'thing', 'Rome', 'Visit', 'Vatican', 'Museums', 'The', 'Vatican', 'City', 'home', 'one', 'famous', 'museum', 'world', 'Vatican', 'Museums', 'The', 'museum', 'house', 'many', 'artistic', 'masterpiece', 'historical', 'artifact', 'date', 'back', 'ancient', 'time', 'You', 'find', 'large', 'collection', 'sculpture', 'painting', 'tapestries', 'The', 'Sistine', 'Chapel', 'one', 'important', 'part', 'museum', 'contain', 'Michelangelo', 'famous', 'fresco', 'ceiling', 'The', 'museum', 'also', 'house', 'artistic', 'masterpiece', 'Raphael', 'Madonna', 'di', 'Foligno', 'Bern']
recognize_entities(text)