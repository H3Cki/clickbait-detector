import plotly.express as px
import plotly.graph_objects as go
import pandas
import text_preprocessing as tp
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import logging
import os
from collections.abc import Iterable

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

clickbait = []
nonclickbait = []

paths = [
    'datasets/mixed_dataset.csv'
]

logging.info('reading dataframes')
dataframes = [pandas.read_csv(path, sep = ';') for path in paths]
for i, df in enumerate(dataframes):
    clickbait += list(df.loc[df['clickbait'] == 1]['title'])
    nonclickbait += list(df.loc[df['clickbait'] == 0]['title'])
print('clickbait titles:', len(clickbait))
print('nonlickbait titles:', len(nonclickbait))

import random

def cb_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl(9, {random.randint(70,100)}%, 50%)"

def ncb_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl(236, {random.randint(50,70)}%, 50%)"

def default_layout(fig, **kwargs):
    layouts = {}
 
    layouts['wide'] = dict(
        paper_bgcolor='#ECECEC',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1900, height=350, 
        uniformtext_minsize=12, 
        uniformtext_mode='hide',
        margin=dict(l=0, r=0, t=0, b=0, pad=1),
        font=dict(
            family="Courier new",
            size=18),
        legend=dict(
            x=1,
            y=1,
            orientation = "v",
            xanchor = "right",
            yanchor = "top",
            bgcolor = 'rgba(255,255,255,0.4)'
        ),
        yaxis=dict(gridcolor="#dedede")
        
    )
    
    #Narrow
    layouts['narrow'] = dict(
        paper_bgcolor='#ECECEC',
        plot_bgcolor='rgba(0,0,0,0)',
        width=420, height=420, 
        uniformtext_minsize=12, 
        uniformtext_mode='hide',
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        font=dict(
            family="Courier new",
            size=18),
        legend=dict(
            x=1,
            y=1,
            orientation = "v",
            xanchor = "right",
            yanchor = "top",
            bgcolor = 'rgba(255,255,255,0.4)'
        ),
        yaxis=dict(gridcolor="#dedede")
    )
    
    t = kwargs.pop('shape') if kwargs.get('shape') else 'wide'
    default = layouts[t]
    
    layout = {**default, **kwargs}
    fig.update_layout(layout)
    # if layout.get('barmode') == 'overlay':
    #     fig.update_traces(opacity=0.8)
    

def plot_hist(cb_x, ncb_x, plots, pyramid=False):
    
    name = plots['name']
    info = plots.get('total', {})
    print('Plotting hist', name)
    
    cb_hist = go.Histogram(
        name = 'Clickbait',
        histnorm='percent',
        x=cb_x,
        marker_color = COLORS[1],
        orientation='h' if pyramid else 'v'
    )

    ncb_hist = go.Histogram(
        name = 'Nie clickbait',
        histnorm = 'percent',
        x=ncb_x,
        marker_color = COLORS[0],
        orientation='h' if pyramid else 'v'
    )
    

    fig = go.Figure()
    fig.add_trace(cb_hist)
    fig.add_trace(ncb_hist)
    
    default_layout(fig, **info['layout'])
    fig.write_image(f"plots/hist_{name}.png")
    
    return fig


def plot_total(cb_x, cb_y, ncb_x, ncb_y, plots, ):

    name = plots['name']
    info = plots.get('total', {})
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Clickbait',
        y=cb_y,
        x=cb_x,
        marker_color = COLORS[1]
    ))
    fig.add_trace(go.Bar(
        name='Nie clickbait',
        y=ncb_y,
        x=ncb_x,
        marker_color = COLORS[0]
    ))
    
    default_layout(fig, **info['layout'])
    fig.write_image(f"plots/{name}.png")
    return fig

def plot_counts(cb_counts, ncb_counts, plots, negative=True):
    
    name = plots['name']
    logging.info(f'Plotting {name}')
    df_dict = {
        'token' : [],
        'clickbait' : [],
        'nonclickbait': [],
        'total': [],
        'c_percent': [],
        'nc_percent': []
    }
    
    for token in set(cb_counts.keys()).union(set(ncb_counts.keys())):
        n_cb = cb_counts.get(token,0)
        n_ncb = ncb_counts.get(token,0)
        total = n_cb + n_ncb
        if total:
            df_dict['token'].append(token)
            df_dict['clickbait'].append(n_cb)
            df_dict['nonclickbait'].append(n_ncb)
            c_percent = n_cb/(total or 1)
            df_dict['c_percent'].append(c_percent)
            df_dict['nc_percent'].append(1-c_percent)
            df_dict['total'].append(total)
            
    
        
    df = pandas.DataFrame(data=df_dict)
    
    if 'detail' in plots:
        logging.info(f' Plotting {name} - detail')
        info = plots['detail']
        if negative:
            info["barmode"] = "relative"
            if not info.get('sort'):
                info["sort"] = ["nonclickbait"]
        head_n = info['data'].get('n')
        if info['mode'] == 'grouped':
            sorts = info.get('sort', ['total',])
            for sort in sorts:
                df = df.sort_values(by=[sort,], ascending=False).head(head_n)
                if negative:
                    df_dict['clickbait'] = [d * (-1) for d in df_dict['clickbait']]
                fig = go.Figure()
                x = list(df['token'].values)
                
                cb_bar = {  'name' : 'Clickbait',
                            'x' : x,
                            'y' : list(df['clickbait'].values),
                            'customdata' : list(df['c_percent']),
                            'texttemplate' : "%{customdata:%}",
                            'marker_color' : COLORS[1]
                        }
                
                ncb_bar = {  'name' : 'Nie clickbait',
                            'x' : x,
                            'y' : list(df['nonclickbait'].values),
                            'customdata' : list(df['nc_percent']),
                            'texttemplate' : "%{customdata:%}",
                            'marker_color' : COLORS[0]
                        }
                
                bar_infos = [cb_bar, ncb_bar]
                if sort == 'nonclickbait':
                    bar_infos = reversed(bar_infos)

                for bar_info in bar_infos:
                    fig.add_trace(go.Bar(**bar_info,
                        textposition="inside",
                        textangle=0,
                        textfont_color="white",
                    ))
                    
                title = info['layout'].get('title')
                if title:
                    info['layout']['title'] = title.replace('<type> ','')
                if negative:
                    info['layout']['yaxis'] = {
                        'ticktext': [str(abs(y)) for y in df['clickbait'].values]
                    }
                    info['layout']['barmode'] = 'overlay'
                    
                default_layout(fig, **info['layout'])
                #fig.show()
                fig.write_image(f"plots/detail_{name}_sort_{sort}.png")
            
        elif info['mode'] == 'split':
            c = df.sort_values(by=['clickbait',], ascending=False).head(head_n)
            n = df.sort_values(by=['nonclickbait',], ascending=False).head(head_n)
            
            raw_title = info['layout'].get('title')
            if raw_title:
                info['layout']['title'] = raw_title.replace("<type>", '[Clickbait]')
            
            fig = go.Figure()
            x = c['token'].values
            fig.add_trace(go.Bar(
                name='Clickbait',
                y=c['clickbait'].values,
                x=x,
                marker_color = COLORS[1]
            ))
            
            default_layout(fig, **info['layout'])
            fig.write_image(f"plots/detail_{name}_clickbait.png")
            #fig.show()
            
            
            fig = go.Figure()
            x = n['token'].values
            fig.add_trace(go.Bar(
                name='Nie clickbait',
                y=n['nonclickbait'].values,
                x=x,
                marker_color = COLORS[0]
            ))
            if raw_title:
                info['layout']['title'] = raw_title.replace("<type>", '[Nonclickbait]')
            default_layout(fig, **info['layout'])
            fig.write_image(f"plots/detail_{name}_nonclickbait.png")
            #fig.show()

        
    ### show totals dict
    if 'total' in plots:
        logging.info(f' Plotting {name} - total')
        info = plots['total']
        cb = round(sum(df_dict['clickbait'])/len(clickbait),2)
        ncb = round(sum(df_dict['nonclickbait'])/len(nonclickbait),2)
        plot_total(cb, ncb, plots)
        
    
    if plots.get('wc'):
        logging.info(f' Plotting {name} - wc')
        ### save clickbait word cloud
        wcloud = WordCloud(background_color='#ECECEC', width=2800, height=2800).generate_from_frequencies(cb_counts)
        plt.figure( figsize=(2.8,2.8), dpi=400)
        plt.imshow(wcloud.recolor(color_func=cb_color, random_state=3), interpolation='antialiased')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f'images/wordcloud_{name}_cb.png')
        
        ### save nonclickbait word cloud
        wcloud = WordCloud(background_color='#ECECEC', width=2800, height=2800).generate_from_frequencies(ncb_counts)
        plt.figure( figsize=(2.8,2.8), dpi=400)
        plt.imshow(wcloud.recolor(color_func=ncb_color, random_state=3), interpolation='antialiased')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f'images/wordcloud_{name}_ncb.png')
    
    #return fig



def visualize():
    
    for n in range(1,4):
        plots = {
            "name" : f"{n}_gram_pos_tag_no_stopwords",
            "detail" : {
                "mode" : "grouped",
                "sort" : ['total'],
                "data" : {
                    "n" : 50
                },
                "layout": {
                    #"title" : f"<type> Najczęściej używane frazy części mowy o długości {n}",
                    "xaxis_title": None,
                    "yaxis_title": None,
                    "legend_title": None,
                    "barmode" : "relative",
                    "xaxis" : {
                        'tickangle' : -30
                    }
                    }
            },
            'wc' : False
        }
        
        ta = dict(
            n=n,
            as_text=True,
            tokenizer=tp.pos_tags,
            tokenizer_args=dict(
                preserve_stopwords=False,
                tokenizer_args=dict(tag_only=True)
            )
        )
        clickbait_counts    = tp.bulk_count_tokens(clickbait, tokenizer=tp.ngram, tokenizer_args=ta)
        nonclickbait_counts = tp.bulk_count_tokens(nonclickbait, tokenizer=tp.ngram, tokenizer_args=ta)
        plot_counts(clickbait_counts, nonclickbait_counts, plots=plots, negative=False)

 
    for n in range(1,4):
     
        plots = {
            "name" : f"{n}_gram_no_stopwords",
            "detail" : {
                "mode" : 'grouped',
                "data" : {
                    "n" : 35
                },
                "layout": {
                    #"title" : f"<type> Najczęściej używane frazy o długości {n}, bez wyrazów typu stop word.",
                    "xaxis_title": None,
                    "yaxis_title": None,
                    "legend_title": None,
                    "barmode" : "relative",
                    "xaxis" : {
                        'tickangle' : -30
                    }
                },
            },
            'wc' : True
        }
        clickbait_counts    = tp.bulk_count_tokens(clickbait[:100], tokenizer=tp.ngram, tokenizer_args={'n': n, 'as_text': True, 'tokenizer_args':{'preserve_stopwords': False}})
        nonclickbait_counts = tp.bulk_count_tokens(nonclickbait[:100], tokenizer=tp.ngram, tokenizer_args={'n': n, 'as_text': True, 'tokenizer_args':{'preserve_stopwords': False}})
        plot_counts(clickbait_counts, nonclickbait_counts, plots=plots, negative=False)


    for n in range(1,4):
        plots = {
            "name" : f"{n}_gram",
            "detail" : {
                "mode" : f"{'grouped' if n > 1 else 'split'}",
                "data" : {
                    "n" : 17 if n > 1 else 10
                },
                "layout": {
                    #"title" : f"<type> Najczęściej używane frazy o długości {n}, bez wyrazów typu stop word.",
                    "xaxis_title": f"{n}-gram",
                    "yaxis_title": "Occurences",
                    "legend_title": None,
                    "barmode" : "relative"
                },
            },
            'wc' : True
        }

        clickbait_counts    = tp.bulk_count_tokens(clickbait, tokenizer=tp.ngram, tokenizer_args={'n': n, 'as_text': True})
        nonclickbait_counts = tp.bulk_count_tokens(nonclickbait, tokenizer=tp.ngram, tokenizer_args={'n': n, 'as_text': True})
        plot_counts(clickbait_counts, nonclickbait_counts, plots=plots)
    
    
    plots = {
        "name" : "pos_tags_max_desc",
        "total" : {
            "layout": {
                "xaxis_title": "POS tags",
                "yaxis_title": "% of titles",
                "legend_title": None,
                "barmode" : "group",
                'xaxis' : {
                    'categoryorder' : 'max descending',
                    'range' : [-0.5, 18.5],
                    'tickangle' : -90
                    }
            },

        }

    }
    ta = dict(tag_only=True)
    f = plot_hist(tp.histogramX(clickbait, tp.pos_tags, mode='word', tokenizer_args=ta), tp.histogramX(nonclickbait, tp.pos_tags, mode='word', tokenizer_args=ta), plots)
   
    
    # #CAPITAL HISTOGRAMS
    
    plots = {
        "name" : "capital_letters",
        "total" : {
            "layout": {
                #"title" : "Number of capital letters",
                "yaxis_title": None,#"% of titles",
                "xaxis_title": None,#"Udział wielkich liter",
                "legend_title": None,
                'shape': 'narrow'
            }
        }
    }
    plot_hist(tp.histogramX(clickbait, tp.capital, mode='num', tokenizer_args={'numeric': True, 'letter_only': True}), tp.histogramX(nonclickbait, tp.capital, mode='num', tokenizer_args={'numeric': True, 'letter_only': True}), plots)

    
    plots = {
        "name" : "capital_words",
        "total" : {
            "layout": {
                #"title" : "Number of capital letters",
                "yaxis_title": "% of titles",
                "xaxis_title": "Share of words with capital letter",
                "legend_title": None,
                'shape': 'narrow'
            }
        }
    }
    plot_hist(tp.histogramX(clickbait, tp.capital, mode='num', tokenizer_args={'numeric': True, 'letter_only': False}), tp.histogramX(nonclickbait, tp.capital, mode='num', tokenizer_args={'numeric': True, 'letter_only': False}), plots)
    

    plots = {
        "name" : "full_capital_words_postags",
        "total" : {
            "layout": {
                #"title" : "Number of fully capital words",
                "yaxis_title": "% of titles",
                "xaxis_title": "Share of capital words",
                "legend_title": None,
                'shape': 'narrow'
            }
        }
    }
    ta = dict(full_capital=True, postags=True, numeric=True)
    plot_hist(tp.histogramX(clickbait, tp.capital, mode='num', tokenizer_args=ta), tp.histogramX(nonclickbait, tp.capital, mode='num', tokenizer_args=ta), plots)
 
    
    
    # #WORD COUNT AND LENGTH HISTOGRAMS
    
    plots = {
        "name" : "word_lengths",
        "total" : {
            "layout": {
                #"title" : "Average word length in titles",
                "yaxis_title": "% of titles",
                "xaxis_title": "Average token length",
                "legend_title": None,
                "barmode" : "group",
                'xaxis' : {'range' : [2,10]},
                'shape' : 'narrow'
            }
        }
    }
    ta = dict(average=True)
    plot_hist(tp.histogramX(clickbait, tp.word_lengths, mode='num', tokenizer_args=ta), tp.histogramX(nonclickbait, tp.word_lengths, mode='num', tokenizer_args=ta), plots)
    

    plots = {
        "name" : "word_n",
        "total" : {
            "layout": {
                #"title" : "Number of words in titles",
                "yaxis_title": None,#"% of titles",
                "xaxis_title": None,#"Ilość tokenów",
                "legend_title": None,
                "barmode" : "group",
                'shape' : 'narrow',
                'xaxis' : {'range' : [0,50]},
            }
        }
    }
    plot_hist(tp.histogramX(clickbait, tp.word_n, mode='num'), tp.histogramX(nonclickbait, tp.word_n, mode='num'), plots)


    
    # #SPECIFIC WORDS HISTOGRAMS
    
    plots = {
        "name" : "stopwords",
        "total" : {
            "layout": {
                #"title" : "Stop words per title",
                "xaxis_title": None,#"Udział wyrazów typu stop word",
                "yaxis_title": None,#"% of titles",
                "legend_title": None,
                'xaxis' : {'range' : [-0.05, 0.65]},
                'shape' : 'narrow'
            }
        }
    }
    ta = dict(numeric=True)
    plot_hist(tp.histogramX(clickbait, tp.stopwords, mode='num', tokenizer_args=ta), tp.histogramX(nonclickbait, tp.stopwords, mode='num', tokenizer_args=ta), plots)
    
    
    plots = {
        "name" : "contractions",
        "total" : {
            "layout": {
                #"title" : "Contractions per title",
                "xaxis_title": "Share of contractions",
                "yaxis_title": "% of titles",
                "legend_title": None,
                'xaxis' : {'range' : [-0.05, 0.25]},
                'shape' : 'narrow'
            }
        }
    }
    ta = dict(numeric=True)
    plot_hist(tp.histogramX(clickbait, tp.contractions, mode='num', tokenizer_args=ta), tp.histogramX(nonclickbait, tp.contractions, mode='num', tokenizer_args=ta), plots)
   

    plots = {
        "name" : "symbols",
        "total" : {
            "layout": {
                #"title" : "Symbols per title",
                "xaxis_title": "Share of symbols",
                "yaxis_title": "% of titles",
                "legend_title": None,
                'shape' : 'narrow',
                'xaxis' : {'range' : [-0.05, 0.75]},
            }
        }
    }
    ta = dict(numeric=True)
    plot_hist(tp.histogramX(clickbait, tp.symbols, mode='num', tokenizer_args=ta), tp.histogramX(nonclickbait, tp.symbols, mode='num', tokenizer_args=ta), plots)
    
    
    plots = {
        "name" : "pronouns",
        "total" : {
            "layout": {
                #"title" : "Symbols per title",
                "xaxis_title": None,#"Udział zaimków",
                "yaxis_title": None,#"% of titles",
                "legend_title": None,
                'shape' : 'narrow',
                'xaxis' : {'range' : [-0.05, 0.55]},
            }
        }
    }
    ta = dict(numeric=True)
    plot_hist(tp.histogramX(clickbait, tp.pronouns, mode='num', tokenizer_args=ta), tp.histogramX(nonclickbait, tp.pronouns, mode='num', tokenizer_args=ta), plots)
    
    
    plots = {
        "name" : "numbers",
        "total" : {
            "layout": {
                #"title" : "Numbers per title",
                "yaxis_title": "% of titles",
                "xaxis_title": "Share of numbers",
                "legend_title": None,
                'shape' : 'narrow',
                'xaxis' : {'range' : [-0.05, 0.45]},
            }
        }
    }
    ta = dict(numeric=True)
    plot_hist(tp.histogramX(clickbait, tp.numbers, mode='num', tokenizer_args=ta), tp.histogramX(nonclickbait, tp.numbers, mode='num', tokenizer_args=ta), plots)
    

    
 
    
    
    # ### ####### ###
    # ### DETAILS ###
    # ### ####### ###
    plots = {
        "name" : "full_capital_pos",
        "detail" : {
            "mode" : "split",
            "data" : {
                "n" : 10
            },
            "layout": {
                #"title" : "<type> Rozkład wyrazów składających się wyłącznie z wielkich liter",
                "xaxis_title": "POS tags of capital words",
                "yaxis_title": "Occurences",
                "legend_title": None,
                "barmode" : "relative"
            }
            
        },
        
        "wc" : True
    }
    
    clickbait_counts    = tp.bulk_count_tokens([tp.capital(title, full_capital=True, postags=True) for title in clickbait])
    nonclickbait_counts = tp.bulk_count_tokens([tp.capital(title, full_capital=True, postags=True) for title in nonclickbait])
    plot_counts(clickbait_counts, nonclickbait_counts, plots=plots)
    
    plots = {
        "name" : "full_capital",
        "detail" : {
            "mode" : "grouped",
            "data" : {
                "n" : 10
            },
            "layout": {
                #"title" : "<type> Rozkład wyrazów składających się wyłącznie z wielkich liter",
                "xaxis_title": "Capital word",
                "yaxis_title": "Occurences",
                "legend_title": None,
                "barmode" : "relative"
            }
            
        },
        
        "wc" : True
    }
    
    clickbait_counts    = tp.bulk_count_tokens([tp.capital(title, full_capital=True) for title in clickbait])
    nonclickbait_counts = tp.bulk_count_tokens([tp.capital(title, full_capital=True) for title in nonclickbait])
    plot_counts(clickbait_counts, nonclickbait_counts, plots=plots)
    
    
    plots = {
        "name" : "pos_tags",
        "detail" : {
            "mode" : "grouped",
            "data" : {
                "n" : 17
            },
            "layout": {
                #"title" : "<type> Distribution of most frequent pos tags",
                "xaxis_title": None,#"Tag części mowy",
                "yaxis_title": None,#"Occurences",
                "legend_title": None,
                "barmode" : "group"
            },
        },
        
        "wc" : True
    }
    clickbait_counts    = tp.bulk_count_tokens(clickbait, tokenizer=tp.pos_tags, tokenizer_args={'tag_only': True})
    nonclickbait_counts = tp.bulk_count_tokens(nonclickbait, tokenizer=tp.pos_tags, tokenizer_args={'tag_only': True})
    plot_counts(clickbait_counts, nonclickbait_counts, plots=plots)
    
    
    plots = {
        "name" : "stopwords",
        "detail" : {
            "sort" : ['total'],
            "mode" : "grouped",
            "data" : {
                "n" : 17
            },
            "layout": {
                #"title" : "<type> Distribution of stop words",
                "xaxis_title": "Stop word",
                "yaxis_title": "Occurences",
                "legend_title": None,
                "barmode" : "relative"
            }
            
        },
        
        "wc" : True
    }

    clickbait_counts    = tp.bulk_count_tokens(clickbait, tokenizer=tp.stopwords)
    nonclickbait_counts = tp.bulk_count_tokens(nonclickbait, tokenizer=tp.stopwords)
    plot_counts(clickbait_counts, nonclickbait_counts, plots=plots, negative=False)
 
    plots = {
        "name" : "contractions",
        "detail" : {
            "mode" : "grouped",
            "sort": ['total'],
            "data" : {
                "n" : 15
            },
            "layout": {
                #"title" : "<type> Distribution of most used word contractions",
                "xaxis_title": "Contraction",
                "yaxis_title": "Occurences",
                "legend_title": None,
                "barmode" : "relative",
                "xaxis" : {
                            'tickangle' : -90
                        }
            }
        },
        "wc" : True
    }
    
    clickbait_counts    = tp.bulk_count_tokens(clickbait, tokenizer=tp.contractions)
    nonclickbait_counts = tp.bulk_count_tokens(nonclickbait, tokenizer=tp.contractions)
    plot_counts(clickbait_counts, nonclickbait_counts, plots=plots, negative=False)

    for i in range(2,4):
        c = []
        for title in clickbait:
            pos_tags = tp.ngram(title, n=i, tokenizer=tp.pos_tags)
            c += pos_tags
        n = []
        for title in nonclickbait:
            pos_tags = tp.ngram(title, n=i, tokenizer=tp.pos_tags)
            n += pos_tags
        
        len_c = len(c)
        len_n = len(n)
        set_c = set(c)
        set_n = set(n)
        len_set_c = len(set_c)
        len_set_n = len(set_n)
        print(f'Ilość unikalnych {i}-gramów części mowy:\nclickbait: {len_set_c} | {len_set_c / len_c}\nnonclickbait: {len_set_n} | {len_set_n / len_n}')
 
    
    plots = {
        "name" : "pronouns",
        "detail" : {
            "mode" : "grouped",
            "data" : {
                "n" : 17
            },
            "sort" : ['total'],
            "layout": {
                #"title" : "<type> Distribution of most used word pronouns",
                "xaxis_title": "Pronoun",
                "yaxis_title": "Occurences",
                "legend_title": None,
                "barmode" : "relative",
                "xaxis" : {
                        'tickangle' : -90
                }
            }
        },
        "wc" : True
    }
    
    clickbait_counts    = tp.bulk_count_tokens(clickbait, tokenizer=tp.pronouns)
    nonclickbait_counts = tp.bulk_count_tokens(nonclickbait, tokenizer=tp.pronouns)
    plot_counts(clickbait_counts, nonclickbait_counts, plots=plots, negative=False)


    plots = {
        "name" : "numbers",
        "detail" : {
            "mode" : "grouped",
            "sort" : ['total',],
            "data" : {
                "n" : 15
            },
            "layout": {
                #"title" : "<type> Distribution of most used symbols",
                "xaxis_title": "Number",
                "yaxis_title": "Occurences",
                "legend_title": None,
                "barmode" : "relative",
                "xaxis" : { "type": 'category' }
            }
        },
        
        "wc" : True
    }
    
    
    
    clickbait_counts    = tp.bulk_count_tokens(clickbait, tokenizer=tp.numbers)
    nonclickbait_counts = tp.bulk_count_tokens(nonclickbait, tokenizer=tp.numbers)
    clickbait_counts.pop('10')
    nonclickbait_counts.pop('10')
    plot_counts(clickbait_counts, nonclickbait_counts, plots=plots, negative=False)
    
 
    
    plots = {
        "name" : "symbols",
        "detail" : {
            "mode" : "grouped",
            "sort" : ['total'],
            "data" : {
                "n" : 17
            },
            "layout": {
                #"title" : "<type> Distribution of most used symbols",
                "xaxis_title": "Symbol",
                "yaxis_title": "Occurences",
                "legend_title": None,
                "barmode" : "relative",
            }
        },
        
        "wc" : True
    }
    
    
    
    clickbait_counts    = tp.bulk_count_tokens(clickbait, tokenizer=tp.symbols)
    nonclickbait_counts = tp.bulk_count_tokens(nonclickbait, tokenizer=tp.symbols)
    plot_counts(clickbait_counts, nonclickbait_counts, plots=plots, negative=False)
    

if __name__ == '__main__':
    visualize()

