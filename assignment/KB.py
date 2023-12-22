import time

import requests

wikidata_values = {"PERSON": ["Q215627"], "NORP": ["Q231002", "Q111252415"], "ORG": ["Q43229"],
                   "GPE": ["Q6256", "Q7275", "Q515"], "LOC": ["Q2221906"], "FAC": ["Q13226383"],
                   "PRODUCT": ["Q2424752"], "WORK_OF_ART": ["Q838948"], "LAW": ["Q7748"], "EVENT": ["Q1656682"],
                   "LANGUAGE": ["Q34770"], "DATE": ["Q205892"], "PERCENT": ["Q11229"], "MONEY": ["Q1368"],
                   "QUANTITY": ["Q309314"], "ORDINAL": ["Q923933"], "CARDINAL": ["Q11563"], "TIME": ["Q11471"]}


def generate_conditions(entity_type: str):
    if not entity_type or entity_type not in wikidata_values.keys():
        return ""

    values = wikidata_values.get(entity_type)
    conditions = []
    for i, value in enumerate(values):
        statement = f"statement{i}"
        condition = f"{{?item p:P31 ?{statement}. ?{statement} (ps:P31/(wdt:P279*)) wd:{value}.}}"
        conditions.append(condition)

    return ' UNION '.join(f"{c}" for c in conditions)


def generate_sparql(entity_name, entity_type):
    cond = generate_conditions(entity_type)
    service = "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". }"
    query_sentence = f"SELECT DISTINCT ?item WHERE {{ {cond} ?item rdfs:label \"{entity_name}\"@en. {service} }}"
    return query_sentence


def wikidata_query(entity_name, entity_type=None):
    url = "https://query.wikidata.org/sparql"
    query = generate_sparql(entity_name, entity_type)
    # set HTTP headers, including User-Agent and Accept
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json'
    }

    params = {
        'query': query,
        'format': 'json'
    }
    # send requests
    response = requests.get(url, headers=headers, params=params)
    entity_ids = []
    # check status code
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', {}).get('bindings', [])

        for result in results:
            item_id = result.get('item', {}).get('value', '')
            entity_ids.append(item_id)
            # item_label = result.get('itemLabel', {}).get('value', '')
            # print(f"Item ID: {item_id}.")
    else:
        print(f"Error {response.status_code}: {response.text}")

    return entity_ids


if __name__ == "__main__":
    res = wikidata_query("Muslim", "NORP")
    print(res)

    time.sleep(1)
    print("______________")

    res = wikidata_query("Italy")
    print(res)
