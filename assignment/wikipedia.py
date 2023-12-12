import requests
from bs4 import BeautifulSoup


def get_wikipedia_page_content(url):
    # 发送 GET 请求获取页面内容
    response = requests.get(url)

    # 检查请求是否成功
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup)
        # 找到页面内容的标签，这取决于维基百科页面的结构
        # 以下是以类名为 "mw-parser-output" 的 <div> 标签为例
        # content_div = soup.find('div', {'class': 'mw-parser-output'})
        # print(content_div)
        content_div = soup.find('div', {'class': 'mw-body-content'})
        # 找到该 div 元素下的所有 <p> 元素
        paragraphs = content_div.find_all('p')

        # 提取每个 <p> 元素的文本内容
        paragraph_texts = [paragraph.get_text() for paragraph in paragraphs]

        return paragraph_texts
        # print(content_div)
        # 提取文本内容
        # if content_div:
        #     content_text = content_div.get_text()
        #     return content_text
        # else:
        #     print("未找到内容的相关标签")
        #     return None
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None


# 维基百科页面的示例链接
wikipedia_url = "https://en.wikipedia.org/wiki/apple"

# 获取并解析页面内容
page_content = get_wikipedia_page_content(wikipedia_url)

# 打印页面内容
for p in page_content:
    print(p.replace("\n", ""))
