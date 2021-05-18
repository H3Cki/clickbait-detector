import requests
import json
import pandas
import text_preprocessing as tp


search_params = ['qwertyuiopasdfghjklzxcvbnm']
base_url = 'https://en.wikinews.org/w/api.php?action=query&format=json&list=allpages&aplimit=max'
bad_titles = []


def query_wiki(apcontinue=None):
    if apcontinue:
        url = base_url + f'&apcontinue={apcontinue}'
    else:
        url = base_url
    response = requests.get(url).json()
    return response


def get_response_titles(resp):
    global bad_titles
    titles = []
    for article in resp['query']['allpages']:
        if tp.tokenize(article['title'], preserve_symbols=False, preserve_numbers=False):
            title = tp.wikispecial(tp.remove_multiquotes(tp.remove_sep(article['title'])))
            titles.append(title)
        else:
            bad_titles.append(article['title'])
    return titles


def get_titles(limit):
    articles = []
    apcontinue = None
    i = 1
    while len(articles) < limit:
        
        try:
            q = query_wiki(apcontinue)
        except:
            break
        
        for new_article in get_response_titles(q):
            if not articles:
                articles.append(new_article)
            else:
                prev_10 = len(articles)-10 if len(articles) > 10 else 0
                unique = True
                for i, old_article in enumerate(articles[prev_10:]):
                    if tp.similar(new_article, old_article):
                        articles[prev_10+i] = new_article
                        unique = False
                        break
                if unique:
                    articles.append(new_article)
            
        if len(articles):
            print(len(articles), 'Titles', f'{i}th Query\n', articles[-1][0])
        i += 1
        
        con = q.get('continue')
        if con:
            con = con.get('apcontinue')
            apcontinue = con
            if not con:
                break
        else:
            break
        
    return list(set(articles))


def create_dataset(limit):
    titles = get_titles(limit)
    
    titles = {
        'title': titles,
        'clickbait': [0 for _ in range(len(titles))]
    }
    
    df = pandas.DataFrame(data=titles)
    df.to_csv('raw_datasets/nonclickbait_wikinews_titles.csv',index=False, sep = ';')
    # with open('bad_titles.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(bad_titles))


if __name__ == "__main__":
    create_dataset(100000000000000000000)
    
    