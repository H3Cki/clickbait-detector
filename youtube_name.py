from apiclient.discovery import build
import logging
from datetime import datetime
import text_preprocessing as tp
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)


USE_CASES = {
    'user' : 'forUsername',
    'channel' : 'id'
}


OLDEST_VID_DATE = datetime.strptime('2015-01-01', '%Y-%m-%d')    


class ChannelNotFoundException(Exception):
    pass


class APIExhaustedException(Exception):
    pass


class YouTube:
    def __init__(self, api_keys=None):
        self.keys = api_keys
        self.keys_idx = -1
        self.build()
        
        
    def build(self):
        self.keys_idx += 1
        if self.keys_idx >= len(self.keys):
            raise APIExhaustedException('No active API key remaining.')
        
        api_key = self.keys[self.keys_idx]
        self.yt = build('youtube', 'v3', developerKey=api_key)
    
    
    def get_channel_name(self, url, limit=None, keys=None, date=None, is_clickbait=None):
        use_case = self.parse_url(url)['uc']
        logging.log(logging.INFO, "Getting playlist id")
        res = self.yt.channels().list(**use_case, part='contentDetails,snippet').execute()
        uploads_playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        channel_name = res['items'][0]['snippet']['title']
        return channel_name
 
    
    def find_custom_channel_id(self, customUrl):
        res = self.yt.search().list(part="snippet", maxResults=50, q=customUrl).execute()
        for item in res['items']:
            if item['id']['kind'] == 'youtube#channel':
                channel_id = item['id']['channelId']
                channel = self.yt.channels().list(part="snippet", id=channel_id).execute()['items'][0]
                if channel['snippet'].get('customUrl') == customUrl.lower():
                    return channel_id
        return None

        
    def parse_url(self, url, fix=True):
        args = url.replace('.', '').split('youtube')[1].split('/')[1:]
        url_type = args[0]
        value = args[1].split('/')[0].split('?')[0]
        
        use_case = {}
        if url_type not in USE_CASES and fix:
            use_case['old'] = {url_type : value}
            channel_id = self.find_custom_channel_id(value)
            
            if not channel_id:
                raise ChannelNotFoundException(f'Could not find channel {value}')
            else:
                value = channel_id
                url_type = 'channel'
                
        use_case['new'] = {url_type : value}
        use_case['uc'] = {USE_CASES.get(url_type, url_type) : value}
        return use_case
        

    def accessible_url(self, url=None, use_case=None):
        if use_case:
            url_type, value = list(use_case['new'].items())[0]
        elif url:
            use_case = self.parse_url(url)
            url_type, value = list(use_case['new'].items())[0]

        return f'https://www.youtube.com/{url_type}/{value}'


    def fix_url_files(self, *paths):
        for path in paths:
            with open(path, 'r') as f:
                file_urls = f.read().split('\n')
            
            fixed_urls = []
            for url in file_urls:
                if url:
                    new_url =  self.accessible_url(url=url)
                    if new_url:
                        fixed_urls.append(new_url)

            with open(path, 'w') as f:
                f.write('\n'.join(list(set(fixed_urls))))
            

    def create_dataset(self, _input, _output, _check, is_clickbait=1):
        import pandas
        
        with open(_input, 'r') as f:
            urls = f.read().split('\n')
        with open(_check, 'r') as f:
            checked_channels = f.read().split('\n')
        
        
        try:
            _f = pandas.read_csv(_output, sep = ';')
            titles = {
                'title' : list(_f['title'].values),
                'clickbait' : list(_f['clickbait'].values)
            }
        except:
            titles = {
                'title' : [],
                'clickbait' : []
            }

        for i, url in enumerate([url for url in urls if not url in checked_channels and url]):
            logging.log(logging.INFO, f"({round(100*(i)/len(urls))}%) Reading channel {i+1} - {url}.")
            
            while True:
                try:
                    vids = self.get_channel_name(url, keys=('title', 'publishedAt'), date=OLDEST_VID_DATE, is_clickbait=is_clickbait)
                    checked_channels.append(url)
                    break
                except ChannelNotFoundException:
                    pass
                except Exception as e:
                    self.build()
                    logging.warn(f'*** REBUILD YOUTUBE. API KEY {self.keys_idx+1}/{len(self.keys)}')
                    logging.warn(str(e))
                except APIExhaustedException as e:
                    logging.warn(str(e))
                    break
            
            
            no_repeat_vids = []
            for i , vid in enumerate(vids):
                title = tp.remove_sep(vid['title'])
                
                idx = min(i, 10)
                prev_10 = vids[i-idx:i]
                unique = True
                for existing in prev_10:
                    if tp.similar(title, existing):
                        unique = False
                        break
                if unique:
                    no_repeat_vids.append(title)
            
            if len(vids) != len(no_repeat_vids):
                logging.log(f'Removed {len(vids) - len(no_repeat_vids)} similar titles.')
            titles['title'] += no_repeat_vids
            titles['clickbait'] += [is_clickbait for _ in range(len(no_repeat_vids))]
        
        with open(_check, 'w') as f:
            f.write('\n'.join(checked_channels))
        
        df = pandas.DataFrame(data=titles)
        df.to_csv(_output,index=False, sep = ';')


def remove_channel_name(vid):
    name = vid['channel_name']
    if name.lower() in ('react',):
        return vid
    title = vid['title']
    title = re.sub(rf"{name}", ' ', title, flags=re.IGNORECASE)
    title = re.sub(rf" 's", '', title, flags=re.IGNORECASE)
    title = re.sub(" +", ' ', title)
    vid['title'] = title
    return vid


def reset_files(*args):
    for arg in args:
        with open(arg, 'w') as f:
            f.write('')


def create_dataset():
    INPUT_CB = 'youtube/yt_clickbait_channels.txt'
    OUTPUT_CB = 'raw_datasets/clickbait_youtube_titles.csv'
    CHECKED_CB = 'youtube/checked_clickbait_channels.txt'

    INPUT_NCB = 'youtube/yt_nonclickbait_channels.txt'
    OUTPUT_NCB = 'raw_datasets/nonclickbait_youtube_titles.csv'
    CHECKED_NCB = 'youtube/checked_nonclickbait_channels.txt'
    
    with open('youtube/yt_api.txt', 'r') as f:
        api_keys = f.read().split('\n')
    y = YouTube(api_keys)
    y.fix_url_files(INPUT_CB, CHECKED_CB, INPUT_NCB, CHECKED_NCB)

    y.create_dataset(INPUT_CB, OUTPUT_CB, CHECKED_CB, is_clickbait=1)
    y.create_dataset(INPUT_NCB, OUTPUT_NCB, CHECKED_NCB, is_clickbait=0)


if __name__ == "__main__":
    #create_dataset()
    with open('youtube/yt_api.txt', 'r') as f:
        api_keys = f.read().split('\n')
    y = YouTube(api_keys)
    with open('youtube/yt_clickbait_channels.txt', 'r') as f:
        channels = f.read().split('\n')
    namet = []
    for channel in channels:
        name = y.get_channel_name(channel)
        namet.append(f'{name}\n            {channel}')
    namet = list(sorted(namet))
    namet = [f'[{i+1}] ' + n for i,n in enumerate(namet)]
    with open('youtube/yt_channel_biblio.txt', 'w+', encoding='utf-8') as f:
        text = '\n'.join(namet)
        f.write(text)
    
