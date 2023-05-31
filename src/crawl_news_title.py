import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

# Utils
def is_valid_format_date(_date) :
    regex = r'\d{4}-\d{2}-\d{2}'
    return  bool(re.match(regex, _date))

# Crawling Code
def crawl_news_title(key_word, max_page=3991):
    crawl_data = pd.DataFrame(columns=['ID', 'text', 'date', 'url'])

    page = 1
    while page <= max_page:
        url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={key}&sort=0&photo=0&field=0&pd=3&ds=2021.01.01&de=2023.04.30&cluster_rank=93&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from20210101to20230430,a:all&start={page}'.format(key=key_word, page=page)
        headers = {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"}

        response = requests.get(url, headers=headers)
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')
        titles_html = soup.select('.group_news > ul > li > div > div > a')
        info_html = soup.select('.group_news > ul > li > div > div > div > .info_group')
        date_html = soup.select('.group_news > ul > li > div > div > div > .info_group > span')

        new_date = date_html.copy()
        for i in range(len(date_html)):
            if not is_valid_format_date(date_html[i].text.replace('.', '-')[:-1]):
                new_date.remove(date_html[i])
        
        is_naver_index = []
        urls = ['' for j in range(10)]
        for i in range(len(info_html)):
            if '네이버뉴스' in info_html[i].get_text():
                is_naver_index.append(i)
                temp_url = info_html[i].select('a')[1].attrs['href']
                urls[i] = temp_url

        if len(new_date) != len(titles_html) : print('Wrong in key:{key} page:{page}'.format(key=key, page=page))

        for i in is_naver_index:
            id = 'crawling_data_key-'+ key_word + '_idx-'+ str(page) + '.' + str(i)
            input_text = titles_html[i].text
            date = new_date[i].text
            url = urls[i]
            crawl_data.loc[len(crawl_data)] = [id, input_text, date, url]

        page += 10
    
    return crawl_data

def get_category(crawl_df):
    cats = []
    date_with_time = []
    except_idx =[]
    for i in range(len(crawl_df)):
        try:
            url = crawl_df['url'][i]
            headers = {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"}
            response = requests.get(url, headers=headers)
            html = response.content
            soup = BeautifulSoup(html, 'html.parser')

            category = soup.select('.media_end_categorize > a > em')[0]
            category = category.text
            cats.append(category)

            dwt = soup.select('.media_end_head_info_datestamp_bunch > span')
            dwt = dwt[0].text
            date_with_time.append(dwt)
            
        except:
            # print(i)
            except_idx.append(i)
            cats.append('none')
            date_with_time.append('none')


    crawl_df['predefined_news_category'] = cats
    crawl_df['date_with_time'] = date_with_time
    print(except_idx)
    


if __name__=="__main__":
    # key_words = ['종합', '대통령', '한국', '삼성', '이란', '출시', '경기', '트럼프', '감독', '게시판', '신간', '정부', '투자', '최고', '중국']
    key_words = ['뉴스', '지원',  '방송']
    # len(key_words)
    # dfs = []
    i = 1
    for key in key_words:
        crawl_df =  crawl_news_title(key)
        get_category(crawl_df)
        crawl_df.to_csv('/opt/ml/data/crawling_data/' + str(i+15)+'_crawl_data.csv')
        i+=1