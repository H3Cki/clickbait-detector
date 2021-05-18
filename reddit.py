import praw
import pandas

def create_dataset():

    reddit = praw.Reddit(client_id='PoZECjyZT3bVnw',
                        client_secret='GhawY4XyUp1lGekum2BPcSiOS-e1_g',
                        user_agent='clickbait_detector')

    headlines = set()

    subs = [
        'technology',
        'science',
        'politics',
        'news',
        'worldnews'
    ]

    target_n = 200000

    for sub in subs:
        i = 0
        limit = int(round(target_n/len(subs)))
        print(sub)
        for submission in reddit.subreddit(sub).new(limit=None):
            prev_len = len(headlines)
            headlines.add(submission.title)
            if prev_len != len(headlines):
                i += 1
        target_n += limit - i


    headlines = list(headlines)

    titles = {
        'title': headlines,
        'clickbait': [0 for _ in range(len(headlines))]
    }
        
    df = pandas.DataFrame(data=titles)
    df.to_csv('raw_datasets/nonclickbait_multi_reddit.csv',index=False, sep = ';')


if __name__ == '__main__':
    create_dataset()