import requests


def wikidata_query(entity_name):
	endpoint_url = "https://query.wikidata.org/sparql"

	# 构建SPARQL查询语句
	query = """
    SELECT ?item ?itemLabel
    WHERE {
      ?item rdfs:label "%s"@en.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
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
		print(results)
	else:
		print(f"Error {response.status_code}: {response.text}")


# 调用查询函数
wikidata_query("Albert Einstein")
