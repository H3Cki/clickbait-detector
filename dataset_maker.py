import pandas
import text_preprocessing as tp
import random

raw_dataset_dir = 'raw_datasets/'
dataset_dir = 'datasets/'
output_file = 'mixed_dataset.csv'

files = [
    'mix_clickbait_data.csv',
    'clickbait_youtube_titles.csv',
    'nonclickbait_arxivData.csv',
    'nonclickbait_multi_reddit.csv',
    'nonclickbait_youtube_titles.csv',
    'nonclickbait_reddit_worldnews.csv'
]

df_dic = {
    'title' : [],
    'clickbait' : []
}

titles = {
    'clickbait' : [],
    'nonclickbait' : []
}

def create_testset(df):
    clickbait = list(df.loc[df['clickbait'] == 1]['title'])
    nonclickbait = list(df.loc[df['clickbait'] == 0]['title'])

    test_cb = random.sample(clickbait, 1250)
    test_ncb = random.sample(nonclickbait, 1250)

    for title in test_cb:
        clickbait.remove(title)
    for title in test_ncb:
        nonclickbait.remove(title)

    test_dic = dict(
        title = test_cb + test_ncb,
        clickbait = [1 for _ in test_cb] + [0 for _ in test_ncb]
    )

    sub_dic = test_df = dict(
        title = clickbait + nonclickbait,
        clickbait = [1 for _ in clickbait] + [0 for _ in nonclickbait]
    )

    sub_df = pandas.DataFrame(data=sub_dic)
    test_df = pandas.DataFrame(data=test_dic)
    test_df.to_csv('datasets/test_dataset.csv', index=False, sep = ';')

    return sub_df
    

def merge_datasets():
    for i, f in enumerate(files):
        df = pandas.read_csv(raw_dataset_dir + f, sep=";")
        if f.startswith("mix"):
            df = create_testset(df)
        ftype = f.split('_')[0]
        if ftype in titles:
            titles[ftype] += [tp.remove_sep(title) for title in random.sample(list(df['title'].values), len(df['title'].values)) if tp.right_length(title)]
        else:
            for index, row in df.iterrows():
                title = row['title']
                if not tp.tokenize(title, preserve_symbols=False, preserve_numbers=False):
                    continue
                clean = tp.remove_sep(title)
                if int(row['clickbait']) == 1:
                    titles['clickbait'].append(clean)
                else:
                    titles['nonclickbait'].append(clean)

    titles['clickbait'] = list(set(titles['clickbait']))
    titles['nonclickbait'] = list(set(titles['nonclickbait']))
    print(len(titles['clickbait']), len(titles['nonclickbait'])) 
    
    if len(titles['nonclickbait']) > len(titles['clickbait']):
        titles['nonclickbait'] = titles['nonclickbait'][:len(titles['clickbait'])]
    else:
        titles['clickbait'] = random.sample(titles['clickbait'], len(titles['nonclickbait']))

    print(len(titles['clickbait']), len(titles['nonclickbait']))    
    
    df_dic['title'] += titles['clickbait']
    df_dic['clickbait'] += [1 for _ in range(len(titles['clickbait']))]
    df_dic['title'] += titles['nonclickbait']
    df_dic['clickbait'] += [0 for _ in range(len(titles['nonclickbait']))]

    df = pandas.DataFrame(data=df_dic)
    df.to_csv(dataset_dir + output_file, index=False, sep = ';')
    print("DONE")
    return df


if __name__ == "__main__":
    merge_datasets()