import requests

wikidata_values = {"PERSON": ["Q215627"], "NORP": ["Q231002", "Q111252415", "Q7278"], "ORG": ["Q43229"], "GPE": [],
                   "LOC": [], "FAC": [], "PRODUCT": [],
                   "WORK_OF_ART": [], "LAW": [], "EVENT": [], "LANGUAGE": [], "DATE": [], "PERCENT": [], "MONEY": [],
                   "QUANTITY": [], "ORDINAL": [], "CARDINAL": [], "TIME": []}


def generate_conditions(entity_type: str):
    if entity_type not in wikidata_values.keys():
        return ""

    values = wikidata_values.get(entity_type)
    conditions = []
    for i, value in enumerate(values):
        statement = f"statement{i}"
        condition = f"{{?item p:P31 ?{statement}. ?{statement} (ps:P31/(wdt:P279*)) wd:{value}.}}"
        conditions.append(condition)

    return ' UNION '.join(f"{c}" for c in conditions)


def generate_sparql(entity_name, entity_type):
    url = "https://query.wikidata.org/sparql"
    cond = generate_conditions(entity_type)
    service = "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". }"
    query_sentence = f"SELECT DISTINCT ?item WHERE {{ {cond} ?item rdfs:label \"{entity_name}\"@en. {service} }}"
    return query_sentence


def wikidata_query(entity_name, entity_type):
    endpoint_url = "https://query.wikidata.org/sparql"

    # 构建SPARQL查询语句
    query = """
    SELECT ?item ?itemLabel
    WHERE {
      ?item rdfs:label "%s"@en.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """ % entity_name

    # 设置HTTP请求头，包括User-Agent和Accept
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json'
    }

    # 设置查询参数
    params = {
        'query': query,
        'format': 'json'
    }

    # 发送HTTP GET请求
    response = requests.get(endpoint_url, headers=headers, params=params)

    # 检查响应状态码
    if response.status_code == 200:
        # 解析JSON数据
        data = response.json()

        # 处理查询结果
        results = data.get('results', {}).get('bindings', [])

        for result in results:
            item_id = result.get('item', {}).get('value', '')
            item_label = result.get('itemLabel', {}).get('value', '')
            print(f"Item ID: {item_id}, Label: {item_label}")
    # print(results)
    else:
        print(f"Error {response.status_code}: {response.text}")


# 调用查询函数
# wikidata_query("Albert Einstein")
res = generate_conditions("NORP")
print(res)

res = generate_sparql("Italian", "NORP")
print(res)
