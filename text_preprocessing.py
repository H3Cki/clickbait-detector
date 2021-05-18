import re
import nltk
import unidecode
import numpy as np
import json
import os
import logging
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob


LEMMER = WordNetLemmatizer()
STEMMER = SnowballStemmer('english')

# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')

# STOPWORDS = {'himself', 'hers', 'as', 'some', 'for', 'an', 'such', 'where', 'its', 'my', 'you', 'but', 'through', 'these', 'the', 'had', 'was', 'have', 'by', 'again', 'from', 'him', 'until', 'there', 'further', 'own', 'before', 'doing', 'during', 'each', 'themselves', 'those', 'our', 'having', 'below', 'over', 'do', 'that', 'into', 'me', 'a', 'are', 'ours', 'because', 'too', 'few', 'then', 'should', 'after', 'to', 'did', 'being', 'yours', 'so', 'itself', 'can', 'ourselves', 'what', 'down', 'her', 'with', 'whom', 'been', 'herself', 'it', 'ma', 'out', 'we', 'not', 'against', 'any', 'up', 'yourself', 'y', 'theirs', 'm', 'when', 'under', 'most', 'were', 'your', 'am', 'yourselves', 'while', 'on', 'this', 'of', 'why', 'does', 'about', 'who', 'is', 'at', 'here', 'will', 'them', 'between', 'than', 'or', 'i', 'nor', 'how', 'no', 'o', 'which', 'other', 'once', 'only', 'same', 'they', 'she', 'very', 'his', 'their', 'might', 'all', 'and', 'both', 'be', 'off', 'has', 'if', 'myself', 'more', 'just', 'he', 'above', 'in', 'now'}

STOPWORDS = {'so', 'here', 'about', 'as', 'ma', 'if', 'because', 'that', 'this', 'of', 'in', 'been', 'ain', 'an', 'its', 'these', 'by', 'needn', 'or', 'which', 'under', 'nor', 'over', 'mightn', 'y', 'why', 'the', 's', 'having', 'just', 'where', 'll', 'same', 'below', 've', 'whom', 'doesn', 'is', 'wasn', 'up', 'shan', 'while', 'into', 'for', 'most', 'on', 'not', 'down', 'didn', 'before', 'both', 'wouldn', 'through', 'do', 'how', 'has', 'mustn', 'isn', 'have', 'am', 'd', 'hadn', 'couldn', 'all', 'be', 'being', 'some', 'again', 'after', 'were', 'doing', 'very', 'hasn', 'with', 'off', 'other', 'only', 'should', 'shouldn', 'from', 'against', 'at', 
'more', 'aren', 'during', 'are', 'weren', 'further', 'once', 'above', 'those', 'than', 'but', 'o', 'own', 're', 'few', 'then', 'between', 't', 'does', 'now', 'can', 'to', 'will', 'each', 'did', 'who', 'and', 'had', 'won', 'was', 'haven', 'too', 'until', 'don', 'when', 'a', 'any', 'out', 'there', 'm', 'such', 'what', 'no'}

CONTRACTIONS = {'who’d', 'you’ve', 'aren’t', "you'll", "you've", "won't", 'they’ll', "shan't", "ain't", 'i’ve', "i've", 'i’d', 'what’ll', 'shan’t', "couldn't", "you're", "weren't", 'haven’t', 'who’re', "doesn't", 'isn’t', "it's", 'she’s', 'mightn’t', "wouldn't", 'there’s', 'who’ve', "aren't", 'where’s', 'what’ve', 'we’re', 'what’s', 'weren’t', "hasn't", "didn't", 'he’s', "haven't", 'won’t', "that'll", 'that’s', "mightn't", 'i’d', 'i’m', 'can’t', 'wouldn’t', 'he’ll', 'who’s', "you'd", 'you’d', "don't", 'didn’t', 'you’ll', "shouldn't", "should've", "isn't", 'they’ve', 'hadn’t', 'couldn’t', "she's", 'what’re', 'he’d', 'don’t', 'hasn’t', 'shouldn’t', 'who’ll', 'they’re', 'mustn’t', "wasn't", 'she’d', 'doesn’t', 'you’re', "needn't", 'they’d', "mustn't", 'i’ll', 'we’d', "hadn't", 'she’ll', 'let’s', 'we’ve'}


PRONOUNS = {'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'our', 'ours', 'ourselves', 'yourselves', 'they', 'them', 'their', 'theirs', 'themselves'} #skipped and us


    
for c in list(CONTRACTIONS):
    syms = ("’", "'", "`")
    for sym in syms:
        if sym in c:
            for sym_2 in syms:
                rplcd = c.replace(sym, sym_2)
                #rplcd2 = c.replace(sym, '')
                if c.split(sym)[0] in PRONOUNS:
                    PRONOUNS.add(rplcd)
                    #PRONOUNS.add(rplcd2)
                CONTRACTIONS.add(rplcd)
                #CONTRACTIONS.add(rplcd2) 

#set(nltk.corpus.stopwords.words('english')).difference(CONTRACTIONS).difference(PRONOUNS)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#TOKENIZER DECORATOR

def require_tokens(decorator_tokenizer=None, **decorator_tokenizer_args):
    def tokenizer_getter(func):
        def token_maker(*args, **kwargs):
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, str) and not kwargs.get('skip_tokenize'):
                    _tokenizer = kwargs.get('tokenizer', decorator_tokenizer)
                    if isinstance(_tokenizer, str):
                        _tokenizer = TOKENIZERS[_tokenizer]
                    _tokenizer_args = {**decorator_tokenizer_args, **kwargs.get('tokenizer_args',{})}   
                    if _tokenizer:
                        args[i] = _tokenizer(arg, **_tokenizer_args) or []
                    else:
                        args[i] = basic_tokenize(arg, **_tokenizer_args) or []            
            return func(*args, **kwargs)
        return token_maker
    return tokenizer_getter

def stem(tokens):
    for i, token in enumerate(tokens):
        _t = LEMMER.lemmatize(token)
        if _t == token:
            _t = STEMMER.stem(token)
        tokens[i] = _t
    return tokens

def remove_links(tokens):
    return [token for token in tokens if not token.startswith('http://') and not token.startswith('https://')]

def tokenize(text, unicode=True, 
             preserve_case=False, 
             preserve_symbols=True, 
             preserve_stopwords=True, 
             preserve_contractions=True, 
             preserve_numbers=False, 
             remove_all=False, 
             **kwargs):
    if isinstance(text, str):
        text = re.sub(r"(?<=[A-Z])\.", '', text)
        if unicode:
            text = unidecode.unidecode(text)
        tokens = nltk.tokenize.TweetTokenizer(preserve_case=preserve_case, reduce_len=False).tokenize(text)
    else:
        tokens = text
    if not preserve_symbols or remove_all:
        tokens = remove_symbols(tokens, **kwargs)
    if not preserve_stopwords or remove_all:
        tokens = remove_stopwords(tokens, **kwargs)
    if not preserve_contractions or remove_all:
        tokens = remove_contractions(tokens, **kwargs)
    if not preserve_numbers or remove_all:
        tokens = remove_numbers(tokens, **kwargs)
    tokens = remove_links(tokens)
    #print(f"Tokenized '{text}' into : {tokens}")
    return tokens


def basic_tokenize(text):
    return text.split(' ')


@require_tokens(tokenize, preserve_symbols=False)
def ngram(tokens, n=2, as_text=False, **kwargs):
    grams = nltk.ngrams(tokens, n)
    if as_text:
        grams = list(map(' '.join, grams))
    return grams


@require_tokens(tokenize, preserve_symbols=False, preserve_numbers=False)
def word_lengths(tokens, **kwargs):
    lengths = list(map(len, tokens))
    lengths = lengths if not kwargs.get('average') else round(sum(lengths)/(len(lengths) or 1),1)
    return lengths


@require_tokens(tokenize, preserve_symbols=False, preserve_numbers=False)
def word_n(tokens, **kwargs):
    return len(tokens)


@require_tokens(tokenize, preserve_symbols=False, preserve_numbers=False)
def stopwords(tokens, preserve_contractions=True, **kwargs):
    _stop_words = STOPWORDS.union(CONTRACTIONS if preserve_contractions else '')
    filtered = [word for word in tokens if word.lower() in _stop_words]
    return round(len(filtered)/(len(tokens) or 1),1) if kwargs.get('numeric') else filtered


@require_tokens(tokenize)
def remove_stopwords(tokens, preserve_contractions=True, **kwargs):
    _stop_words = STOPWORDS.union(CONTRACTIONS if preserve_contractions else '')
    filtered = [word for word in tokens if word.lower() not in _stop_words]
    return len(filtered)/(len(tokens) or 1) if kwargs.get('numeric') else filtered


@require_tokens(tokenize, preserve_symbols=False, preserve_numbers=False)
def contractions(tokens, **kwargs):
    filtered = [token for token in tokens if token.lower() in CONTRACTIONS]
    return round(len(filtered)/(len(tokens) or 1),1) if kwargs.get('numeric') else filtered


@require_tokens(tokenize)
def remove_pronouns(tokens, **kwargs):
    return [token for token in tokens if token.lower() not in PRONOUNS]


@require_tokens(tokenize, preserve_symbols=False, preserve_numbers=False)
def pronouns(tokens, **kwargs):
    filtered = [token for token in tokens if token.lower() in PRONOUNS]
    return round(len(filtered)/(len(tokens) or 1),1) if kwargs.get('numeric') else filtered


@require_tokens(tokenize)
def remove_contractions(tokens, **kwargs):
    return [token for token in tokens if token.lower() not in CONTRACTIONS]

@require_tokens(tokenize)
def symbols(tokens, **kwargs):
    filtered = re.findall(r"[^A-z0-9 ]", ''.join(tokens))
    return round(len(filtered)/(len(tokens) or 1),1) if kwargs.get('numeric') else filtered


@require_tokens(tokenize)
def remove_symbols(tokens, **kwargs): # remove pure symbols from token list
    cheat_symbols = [' l ',]
    to_remove = []
    if not kwargs.get('words'):
        for token in tokens:
            subbed = re.sub(rf"[^a-zA-Z0-9 ]", '', token) # remove all non-alphabetical and non-numerical symbols
            for cheat_symbol in cheat_symbols:
                subbed = re.sub(rf"[{cheat_symbol}]", '', subbed)
            if not len(subbed):
                to_remove.append(token)
        for token in to_remove:
            tokens.remove(token)
    else:
        new_tokens = []
        for token in tokens:
            token = re.sub(r'([^A-z0-9]+)', '', token)
            if token:
                new_tokens.append(token)
        tokens = new_tokens
    return tokens


@require_tokens(tokenize, preserve_numbers=True)
def numbers(tokens, **kwargs):
    #filtered = [token for token in tokens if re.match(r'^-?\d+(?:\.\d+)?$', token) or kwargs.get('int', False)]
    filtered = [token for token in tokens if token.isdigit() or (len(token.split('.')) == 2 and all([t.isdigit() for t in token.split('.')]))]
    return round(len(filtered)/(len(tokens) or 1),1) if kwargs.get('numeric') else filtered


@require_tokens(tokenize)
def remove_numbers(tokens, **kwargss):
    return [token for token in tokens if not token in numbers(tokens)]


def pos_tags_blob(text, tag_only=True, **kwargs):
    tags = TextBlob(text).tags
    if tag_only:
        tags = [tag for _, tag in tags]
    return tags

@require_tokens(tokenize, preserve_case=False)
def pos_tags(tokens, tag_only=True, **kwargs):
    tags = nltk.pos_tag(tokens)
    if tag_only:
        tags = [tag for _, tag in tags]
    return tags


@require_tokens(tokenize, preserve_case=True, preserve_symbols=False, preserve_numbers=False)
def capital(tokens, full_capital=False, letter_only = True, lower=False, postags=False, **kwargs):
    _tokens = [re.sub(r'\W+', '', token) for token in tokens]
    filtered = []
    if not full_capital:
        if postags:
            raise Exception('Cant postag when full_capital is False.')
        if letter_only:
            for t in _tokens:
                filtered += re.findall(r"[A-Z]", t)
        else:
            filtered = [tokens[i] for (i,token) in enumerate(_tokens) if re.findall(r"[A-Z]", token)]
    else:
        filtered = [tokens[i] for (i,token) in enumerate(_tokens) if not re.sub(r"[A-Z]", "", token)]
        if postags:
            filtered = [pos_tags(token)[0] for token in filtered]
            
    fl = len(filtered)
    if kwargs.get('numeric'):
        if full_capital:
            val = fl / (len(tokens) or 1)
        elif letter_only:
            val = fl / (len(''.join(_tokens)) or 1)
        else:
            val = fl / (len(tokens) or 1)
        return round(val, 1)
    
    if lower:
        filtered = [t.lower() for t in filtered]
    
    return filtered

@require_tokens(tokenize)
def count_tokens(tokens, counts={}, **kwargs):
    for token in tokens:
        if counts.get(token):
            counts[str(token)] += 1
        else:
            counts[str(token)] = 1
    return counts

        
def bulk_count_tokens(token_list, **kwargs):
    counts = {}
    for tokens in token_list:
        if tokens:
            counts = count_tokens(tokens, counts, **kwargs)
    return counts


@require_tokens(tokenize)
def _vocab(tokens):
    return set(tokens)


def vocab(tokens_list):
    v = set()
    if isinstance(tokens_list, (list, tuple)):
        for tokens in tokens_list:
            v = v.union(_vocab(tokens))
    else:
        v = v.union(_vocab(tokens_list))
    return v


@require_tokens(tokenize, preserve_symbols=False, preserve_stopwords=False)
def similarity(tokens_a, tokens_b, **kwargs):
    X_set = set(tokens_a)
    Y_set = set(tokens_b)
    
    l1 = []
    l2 = [] 
    

    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        l1_val = 1 if w in X_set else 0
        l2_val = 1 if w in Y_set else 0
        l1.append(l1_val)
        l2.append(l2_val)
        
    c = 0
    # cosine formula  
    for i in range(len(rvector)): 
            c += l1[i]*l2[i]
    divisor = float((sum(l1)*sum(l2))**0.5)
    return c / divisor if divisor else 0

def similar(a,b,degree=0.75,**kwargs):
    return similarity(a,b,**kwargs) >= degree

@require_tokens(tokenize)
def right_length(text):
    try:
        l = len(text)
        return l > 0 and l < 64
    except:
        return False

def _histogram(iterable, tokenizer, mode='len', cap=None, display='percentage', **kwargs):
    # Modes:
    # len - length - tokenizer response length
    # num - number - single number tokenizer response
    if mode == 'word':
        counts = bulk_count_tokens(iterable, tokenizer=tokenizer, **kwargs)
        X = sorted(counts.keys(), key = lambda _key: counts[_key], reverse=True)
        Y = [counts[key] for key in X]
    else:
        total = []
        for item in iterable:
            tokens = tokenizer(item, **kwargs.get('tokenizer_args', {}))
            if mode == 'len':
                total.append(len(tokens))
            elif mode == 'num' or mode == 'avg':
                total.append(tokens)
            elif mode == 'cum':
                total += tokens
        X = list(range(int(min(total)), int(max(total))+1))
        Y = [100*total.count(x)/len(total) for x in X] if display == 'percentage' else  list(map(total.count, X))
    return X, Y

def histogramX(iterable, tokenizer, mode='len', display='percentage', **kwargs):
    if mode == 'word':
        words = []
        for item in iterable:
            tokens = tokenizer(item, **kwargs.get('tokenizer_args', {}))
            words += tokens
        X = words
    else:
        total = []
        for item in iterable:
            tokens = tokenizer(item, **kwargs.get('tokenizer_args', {}))
            if mode == 'len':
                total.append(len(tokens))
            elif mode == 'num':
                total.append(tokens)
            elif mode == 'cum':
                total += tokens

        X = total
    return X

def histogram_num_feats(iterable):
    size = len(numeric_features(' '))
    avg = [0 for _ in range(size)]

    for text in iterable:
        feats = numeric_features(text)
        avg = [avg[i] + feats[i] for i in range(size)]

    avg = [a/len(iterable) for a in avg]

    return avg

def remove_sep(text):
    return text.replace(';', ',')


def remove_multiquotes(text):
    quote_symbols = ['"', "'"]
    for s in quote_symbols:
        text = re.sub(rf'(?<!{s}){s}(?!:{s})', '', text)
        text = re.sub(rf'(?<={s}){s}', '', text)
    return text


def wikispecial(text):
    return text

def numeric_features(text, normal=True):
    word_nm = word_n(text)
    avg_len = word_lengths(text, average=True)
    stopword_r = stopwords(text, numeric=True)
    symbol_r = symbols(text, numeric=True)
    pronoun_r = pronouns(text, numeric=True)
    feats = normalize([word_nm, avg_len, stopword_r, symbol_r, pronoun_r])
    return feats

def normalize(vec, max_val=None):
    max_val = max_val or max(vec)
    return [v/(max_val or 1) for v in vec]

def array_to_list(lst):
    if isinstance(lst, np.ndarray):
        lst = lst.tolist()
    elif isinstance(lst, (list, tuple)):
        if isinstance(lst[0], np.ndarray):
            for i, v in enumerate(lst):
                lst[i] = v.tolist()
    return lst

def list_to_array(lst):
    if isinstance(lst, (list, tuple)):
        if isinstance(lst[0], (list, tuple)):
            for i, v in enumerate(lst):
                lst[i] = np.asarray(v)
        return np.asarray(lst)
    return lst

def as_array(func):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        return np.asarray(out).astype('float32')
    return wrapper

def clean_text(text, **kwargs):
    return " ".join(tokenize(text, **kwargs))


TOKENIZERS = dict(
    tokenize=tokenize,
    pos_tags=pos_tags,
    stopwords=stopwords,
    symbols=symbols,
    contractions=contractions,
    capital=capital,
    numbers=numbers,
    word_len=word_lengths,
    ngram=ngram,
    numeric=numeric_features,
    pronouns=pronouns
)




class Encoder:
    @staticmethod
    def load(config):
        print(config)
        for c in Encoder.__subclasses__():
            if c.__name__.lower() == config['type'].lower():
                _class = c
                break
        encoder = _class(config=config)
        return encoder

    def __init__(self, **kwargs):
        self.config = kwargs.get('config')
        self.type = None
        self._tokenizer = None
        self.max_len = None
        self.tokenizer_args = {}
        self.load_config(self.config)
        self.tokenizer = TOKENIZERS.get(self._tokenizer, None)
        self.name = f'{self.type}-{self._tokenizer}-{self.max_len}-{"_".join([f"{item[0]}_{item[1]}" for item in self.tokenizer_args.items()])}'
        self.setup()
        
    def load_config(self, config):
        for var in config:
            setattr(self, var, list_to_array(config[var]))
        
    def setup(self):
        pass

    def tokenize(self, text):
        return self.tokenizer(text, **self.tokenizer_args)
    
    def encode(self, text):
        raise Exception('ENCODE NOT IMPLEMENTED')
    

def table(titles, _tokenizer, tokenizer_args={}, read=True, fname=None):
    if fname and read:
        try:
            with open(f'tables/{fname}.json', 'r') as f:
                table = json.load(f)
                logging.info(f'Loaded table {fname}')
                return table
        except:
            pass
    
    logging.info(f'Creating table {fname}')
    
    tokenizer = TOKENIZERS[_tokenizer]#if isinstance(tokenizer, str) else tokenizer
    tokenized_titles = [tokenizer(title, **tokenizer_args) for title in titles]
    vocab = []
    max_len = 0
    for tokens in tokenized_titles:
        if max_len < len(tokens):
            max_len = len(tokens)
        vocab += tokens
    vocab = set(vocab)
    
    offset = 0
    table = {token : i+offset for i,token in enumerate(vocab)}
    
    dct = dict(max_len=max_len, vocab_size=len(vocab)+offset, table=table)
    
    if fname:
        with open(f'tables/{fname}.json', 'w') as f:
            json.dump(dct, f)
    
    logging.info(f'Created table {fname}')
    
    return dct
    
def merge_tables(tables, fname=None):
    vocab = set()
    max_len = -1
    for table in tables:
        vocab = vocab.union(table['table'])
        if table['max_len'] > max_len:
            max_len = table['max_len']

    offset = 1
    table = {token : i+offset for i,token in enumerate(vocab)}
    dct = dict(max_len=max_len, vocab_size=len(vocab)+offset, table=table)
    
    if fname:
        with open(f'tables/{fname}.json', 'w') as f:
            json.dump(dct, f)
    
    logging.info(f'Created table {fname}')
    
    return dct
        
class TableEncoder(Encoder):
    config_vars = (
        'type', '_tokenizer', 'max_len', 'tokenizer_args'
    )
    
    def setup(self):
        self.table_name = self.config["table"]
        with open(f'tables/{self.table_name}.json', 'r') as f:
            self.table = json.load(f)
    
    def encode(self, text):
        tokens = self.tokenizer(text, **self.tokenizer_args)
        encoded = [self.table['table'][token] for token in tokens if token in self.table['table']]
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        elif len(encoded) < self.max_len:
            encoded += [0 for _ in range(self.max_len - len(encoded))]
        return encoded


class CustomEncoder(Encoder):
    config_vars = (
        'type', '_tokenizer', 'max_len', 'tokenizer_args'
    )
    
    def encode(self, text):
        return self.tokenizer(text, **self.tokenizer_args)


print(tokenize("THE BEST VIDEOS OF ALL TIME YOU MUST SEE!! *CRAZY*"))
print(tokenize("THE BEST VIDEOS OF ALL TIME YOU MUST SEE!! *CRAZY*", preserve_stopwords=False, preserve_symbols=False))
