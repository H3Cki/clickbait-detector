import json
import pandas as pd

to_convert = [
    'nonclickbait_arxivData.json'
]


def create_dataset():
    for f in to_convert:
        with open(f'raw_json_datasets/{f}', 'r') as fp:
            data = json.load(fp)
        titles = [d['title'] for d in data]
        is_cb = 1 if f.startswith('clickbait') else 0
        
        data = {
            'title': titles,
            'clickbait': [0 for _ in range(len(titles))]
        }
        
        input(len(titles))
        
        df = pd.DataFrame(data=data)
        df.to_csv(f'raw_datasets/{f.split(".")[0]}.csv', index=False, sep = ';')
        
        
if __name__ == '__main__':
    create_dataset()