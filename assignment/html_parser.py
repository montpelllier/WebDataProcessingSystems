import requests
from bs4 import BeautifulSoup


def parse_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    else:
        print(f"failed, status code: {response.status_code}")
        return None


def get_wikipedia_url(soup):
    content = soup.find('span', {'class': 'wikibase-sitelinkview-link wikibase-sitelinkview-link-enwiki'})
    if not content:
        return None

    content = content.find('span', {'class': 'wikibase-sitelinkview-page'}).find('a')
    wikipedia_url = content.get('href')

    return wikipedia_url


def get_wikidata_brief(soup):
    content = soup.find('div', {'class': "wikibase-entitytermsview-heading-description"})
    if not content:
        return None
    return content.get_text()


def get_wikipedia_page_content(soup):
    # 以下是以类名为 "mw-parser-output" 的 <div> 标签为例
    content_div = soup.find('div', {'class': 'mw-body-content'})
    # 找到该 div 元素下的所有 <p> 元素
    paragraphs = content_div.find_all('p')
    # 提取每个 <p> 元素的文本内容
    paragraph_texts = []
    for paragraph in paragraphs:
        if len(paragraph.get_text()) >= 20:
            paragraph_texts.append(paragraph.get_text())

    return paragraph_texts


if __name__ == "__main__":

    # # 维基百科页面的示例链接
    # wikipedia_url = "https://en.wikipedia.org/wiki/apple"
    # # 获取并解析页面内容
    # page_content = get_wikipedia_page_content(wikipedia_url)
    # # 打印页面内容
    # for p in page_content:
    #     print(p.replace("\n", ""))
    # url = "https://www.wikidata.org/wiki/Q47740"

    # url = "https://www.wikidata.org/wiki/Q2781527"
    url = "http://www.wikidata.org/wiki/Q209878"
    soup = parse_url(url)
    # print(soup)
    if soup:
        wikipedia_url = get_wikipedia_url(soup)
        brief = get_wikidata_brief(soup)
        print(wikipedia_url, brief)
