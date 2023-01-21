from political_nlp import pnlp
import sankey
import topic_mdl
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import re as re
import string
import wordcloud as wc
from nltk.corpus import stopwords
import base64
from os.path import exists
import os

import warnings
warnings.filterwarnings("ignore")


def load_neg_words(filename):
    """
    load list of negative words for negative-ratio sentiment analysis
    :param filename: (str) name of negative words file
    :return: (list) list of negative words
    """

    # open file and read all lines
    f = open(filename, "r")

    # make list of words
    lst = f.readlines()

    # remove new lines for all words
    for i in range(len(lst)):
        lst[i] = lst[i].replace("\n", "")

    # ignore headers
    lst = lst[35:]

    return lst


# make negative words list a global variable
negs = load_neg_words("negative-words.txt")

# saves the topics and their important event info dict as a global variable
topics = {'abortion': [{'x_vline':20, 'x_annotation': 19.6, 'text': 'Roe v Wade Overturned'},
            {'x_vline': 5, 'x_annotation': 4.6, 'text': 'Gonzales v. Planned Parenthood'},
            {'x_vline': 14, 'x_annotation': 13.6, 'text': "Whole Woman's Health v. Hellerstedt"},
            {'x_vline': 18, 'x_annotation': 17.6, 'text': 'June Medical Services v. Russo'},
            {'x_vline':19, 'x_annotation':18.6, 'text':'Texas Six-Week Ban'}],
          'gay marriage': [{'x_vline': 13, 'x_annotation': 12.6, 'text': 'Gay Marriage Legalized Federally'},
            {'x_vline': 1, 'x_annotation': 0.6, 'text': 'Lawrence v. Texas'},
            {'x_vline': 2, 'x_annotation': 1.6, 'text': 'Mass Legalizes Gay Marriage'},
            {'x_vline': 6, 'x_annotation': 5.6, 'text': 'Cali voters approve Proposition 8'},
            {'x_vline': 7, 'x_annotation': 6.6, 'text': 'Matthew Shepard Act'},
            {'x_vline': 8, 'x_annotation': 7.6, 'text': 'Prop 8 deemed unconstitutional'}],
          'marijuana': [{'x_vline': 7, 'x_annotation': 6.6, 'text': 'DOJ Lenient to Medical Marijuana Patients'},
            {'x_vline': 12, 'x_annotation': 11.6, 'text': 'Rohrabacher–Farr amendment'},
            {'x_vline': 16, 'x_annotation': 15.6, 'text': 'CBD legalized'}],
          'immigration': [{'x_vline':0, 'x_annotation':0.6, 'text': 'Homeland Security Act'}, \
                          {'x_vline':10, 'x_annotation':9.6, 'text':'DACA established'}, \
                          {'x_vline': 15, 'x_annotation':14.6, 'text':'Muslim Travel Ban'}]}


def load_topic(start, end, topic, negs):
    """
    get sentiment stats and text data for all years of a topic
    :param start: (int) start year
    :param end: (int) end year
    :param topic: (str) topic of interest
    :param negs: (lst) list of negative words
    :return: nyt_passages: (dict) all article text for nyt with year/text for key/val
                nyp_passages: (dict) all article text for nytpwith year/text for key/val
                df: (dataframe) all years and sentiment stats for specified year
    """

    # create dictionaries for nyt and nyp passages
    nyt_passages = {}
    nyp_passages = {}

    # make bool for first iteration of loop
    first = True

    # for each year in specified range
    for i in range(start, end + 1):

        # update which year is being loaded to user
        print("Loading for year:", i)

        lst = []
        lst.append(i)

        # make pnlp object for nyt year of interest and load text
        nyt = pnlp(f"{topic}-nyt/nyt_{topic}_{i}.json", negs)
        nyt.load_text("nyt")

        # make empty string
        nyt_str = ""

        # add all article text to nyt_str
        for k, v in nyt.text.items():
            new = re.sub('[' + string.punctuation + ']', '', v).lower().strip()
            nyt_str += new

        # add nyt_str to nyt_passages dct
        nyt_passages[i] = nyt_str

        # make pnlp object for nyt year of interest and load text
        nyp = pnlp(f"{topic}-nyp/nyp_{topic}_{i}.json", negs)
        nyp.load_text("nyp")

        # make empty string
        nyp_str = ""

        # add all article text to nyt_str
        for k, v in nyp.text.items():
            new = re.sub('[' + string.punctuation + ']', '', v).lower().strip()
            nyp_str += new

        # add nyt_str to nyt_passages dct
        nyp_passages[i] = nyp_str

        # for first iteration of for loop
        if first == True:

            # make a df of first year of interest
            df = pd.DataFrame(
                {'year': [i], "nyt-pos": [nyt.pos], "nyt-neg": [nyt.neg], "nyt-neu": [nyt.neu], "nyp-pos": [nyp.pos],
                 "nyp-neg": [nyp.neg], "nyp-neu": [nyp.neu], "nyt-neg_ratio": [nyt.neg_ratio],
                 "nyp-neg_ratio": [nyp.neg_ratio]})

        # otherwise, make a new df for each year
        else:
            ndf = pd.DataFrame(
                {'year': [i], "nyt-pos": [nyt.pos], "nyt-neg": [nyt.neg], "nyt-neu": [nyt.neu], "nyp-pos": [nyp.pos],
                 "nyp-neg": [nyp.neg], "nyp-neu": [nyp.neu], "nyt-neg_ratio": [nyt.neg_ratio],
                 "nyp-neg_ratio": [nyp.neg_ratio]})

            # concat new df to first year of interest df
            df = pd.concat([df, ndf])

        # no longer = first iteration
        first = False

    # return nyt, nyp passages and stats df
    return nyt_passages, nyp_passages, df


def wordcloud(nyt_texts, nyp_texts):
    """

    :param nyt_texts: string containing source 1 articles
    :param nyp_texts: string containing source 2 articles
    :return: wordcloud visualizations of most common words from each source
    """

    # making list of stop words to remove
    stops = list(set(stopwords.words('english')))
    stops.append("s")
    stops.append("said")

    # make strings for all nyt and nyp articles of all years
    nytstring = ''
    nypstring = ''

    # add all texts to strings
    for year in range(2002, 2023):
        nytstring += nyt_texts[year]
        nypstring += nyp_texts[year]

    # split strings
    nyt = nytstring.split()
    nyp = nypstring.split()

    # remove stop words
    for word in stops:
        nyt = [value for value in nyt if value != word]
        nyp = [value for value in nyp if value != word]

    # join words
    nyt = ' '.join(nyt)
    nyt = nyt.replace(" s ", "")
    nyp = ' '.join(nyp)
    nyp = nyp.replace(" s ", "")

    # make wordcloud object
    cloud = wc.WordCloud()

    # make wordcloud figure for nyt
    nyt_cloud = cloud.generate(nyt)
    fig1 = go.Figure()

    # set orientation and layout
    fig1.add_trace(go.Image(z=nyt_cloud))
    fig1.update_layout(height=600, width=800, xaxis={'visible': False}, yaxis={'visible': False},
                       title_text='New York Times',
                       paper_bgcolor='rgb(201, 204, 209)')

    # make wordcloud figure for nyp
    nyp_cloud = cloud.generate(nyp)
    fig2 = go.Figure()

    # make wordcloud figure for nyt
    fig2.add_trace(go.Image(z=nyp_cloud))
    fig2.update_layout(height=600, width=800, xaxis={'visible': False}, yaxis={'visible': False},
                       title_text='New York Post',
                       paper_bgcolor='rgb(201, 204, 209)')

    # return nyt and nyp wordcloud figs
    return fig1, fig2


def parallel(start, end, topic, negs):
    """
    make parallel coordinates plot
    :param df: (dataframe) containing all sentiment score avgs data for each year nyt vs nyp
    :param topic: (str) topic of interest
    :param start: (int) start year of interest
    :param end: (int) end year of interest
    :return: parellel_fig: figure
    """
    # load texts for year of interest
    nyt_texts, nyp_texts, df = load_topic(start, end, topic, negs)

    # make nyt df from only nyt columns
    nyt_df = df[['year', 'nyt-pos', 'nyt-neu', 'nyt-neg']]

    # rename columns
    nyt_df.rename(
        columns={'nyt-pos': 'Positive Sentiment', 'nyt-neu': 'Neutral Sentiment',
                 'nyt-neg': 'Negative Sentiment'}, inplace=True)

    # make journal column
    nyt_df['Journal'] = [0] * len(nyt_df)

    # make nyp df from only nyp columns
    nyp_df = df[['year', 'nyp-pos', 'nyp-neu', 'nyp-neg']]

    # rename columns
    nyp_df.rename(
        columns={'nyp-pos': 'Positive Sentiment', 'nyp-neu': 'Neutral Sentiment',
                 'nyp-neg': 'Negative Sentiment'}, inplace=True)

    # make jounral column
    nyp_df['Journal'] = [1] * len(nyp_df)

    # add together nyt and nyp dfs sorting by year
    parallel_df = pd.concat([nyp_df, nyt_df]).sort_values('year')

    # create vis figure
    parallel_fig = px.parallel_coordinates(parallel_df, color='Journal',
                                           dimensions=['Positive Sentiment', 'Neutral Sentiment', 'Negative Sentiment'],
                                           color_continuous_scale=px.colors.diverging.Tealrose,
                                           color_continuous_midpoint=0.5
                                           )

    return parallel_fig


def sank(num_words, year, topic, negs):
    """
    make sankey vis fig
    :param num_words: (int) number of top nouns for sankey to display for nyt and nyp
    :param year: (int) year of interest to show sankey for
    :param topic: (str) topic of interest
    :param negs: (lst) list of loaded negative words
    :return: fig
    """

    # load noun dictionary with counts for specified year for nyp and nyt
    nyp = pnlp(f"{topic}-nyp/nyp_{topic}_{year}.json", negs)
    nyp.load_text("nyp")
    dct1 = nyp.noun_scores
    nyt = pnlp(f"{topic}-nyt/nyt_{topic}_{year}.json", negs)
    nyt.load_text("nyt")
    dct2 = nyt.noun_scores

    # create list for target, source, and vals
    vlst = []
    wlst = []
    clst = []

    # define for user specified desired number of words to show in sankey
    num = num_words

    # iterate through nyt nouns
    for k, v in dct1.items():

        # if not surpassed user-specified desired number of nouns for sankey
        if num >= 0:

            # append targ, src, and val
            vlst.append(f"New York Post {year} : {topic}")
            wlst.append(k)
            clst.append(v)

        # break when surpassed num_words
        else:
            break
        num -= 1

    # define for user specified desired number of words to show in sankey
    num = num_words

    # iterate through nyp nouns
    for k, v in dct2.items():

        # if not surpassed user-specified desired number of nouns for sankey
        if num >= 0:

            # append targ, src, and val
            vlst.append(f"New York Times {year} : {topic}")
            wlst.append(k)
            clst.append(v)

        # break when surpassed num_words
        else:
            break
        num -= 1

    # make df for src, targ, and vals
    df = pd.DataFrame()
    df["src"] = vlst
    df["targ"] = wlst
    df["vals"] = clst

    # return sankey figure
    return sankey.make_sankey(df, "src", "targ", "vals", year, topic)


def stacked(df, topic, topic_dict, negs):
    """
    make stacked scatter vis fig
    :param df: (dataframe) df of all sentiment score averages for each year of a topic byt nyt and nyp
    :param topic_dict: (dict) topic of interest as key, numbers for event line as values
    :param negs: (lst) list of all neg words
    :return: fig
    """

    # make list to append year vals (ysed for nyt  and nyp)
    year = []

    # lists for nyt vader sent vals
    nyt_pos_val = []
    nyt_neg_val = []
    nyt_neu_val = []

    # lists for nyp vader sent vals
    nyp_pos_val = []
    nyp_neg_val = []
    nyp_neu_val = []

    # iterate through sentiment stats df
    for i in range(len(df)):
        # append years
        year.append(df.iloc[i]['year'])

        # append pos, neg, neu sentiments for nyt and nyp
        nyt_pos_val.append(df.iloc[i]['nyt-pos'])
        nyp_pos_val.append(df.iloc[i]['nyp-pos'])
        nyt_neg_val.append(df.iloc[i]['nyt-neg'])
        nyp_neg_val.append(df.iloc[i]['nyp-neg'])
        nyt_neu_val.append(df.iloc[i]['nyt-neu'])
        nyp_neu_val.append(df.iloc[i]['nyp-neu'])

    # make subplots
    stackfig = make_subplots(rows=1, cols=2,
                             subplot_titles=("NY Times", "NY Post"))

    # add stacked scatters for nyt vals in row 1, col 1
    # set all layout parameters
    stackfig.add_trace(go.Scatter(
        x=year, y=nyt_pos_val,
        name='pos score',
        mode='lines',
        line=dict(width=0.5, color='green'),
        stackgroup='one',
        groupnorm='percent'  # sets the normalization for the sum of the stackgroup
    ), row=1, col=1)
    stackfig.add_trace(go.Scatter(
        x=year, y=nyt_neg_val,
        name='neg score',
        mode='lines',
        line=dict(width=0.5, color='red'),
        stackgroup='one'
    ), row=1, col=1)
    stackfig.add_trace(go.Scatter(
        x=year, y=nyt_neu_val,
        name='neu score',
        mode='lines',
        line=dict(width=0.5, color='blue'),
        stackgroup='one'
    ), row=1, col=1)

    # add stacked scatters for nyp vals in row 1, col 2
    # set all layout parameters
    stackfig.add_trace(go.Scatter(
        x=year, y=nyp_pos_val,
        name='pos score',
        mode='lines',
        line=dict(width=0.5, color='green'),
        stackgroup='one',
        groupnorm='percent',  # sets the normalization for the sum of the stackgroup
        showlegend=False
    ), row=1, col=2)
    stackfig.add_trace(go.Scatter(
        x=year, y=nyp_neg_val,
        name='neg score',
        mode='lines',
        line=dict(width=0.5, color='red'),
        stackgroup='one',
        showlegend=False
    ), row=1, col=2)
    stackfig.add_trace(go.Scatter(
        x=year, y=nyp_neu_val,
        name='neu score',
        mode='lines',
        line=dict(width=0.5, color='blue'),
        stackgroup='one',
        showlegend=False
    ), row=1, col=2)

    # update layout
    stackfig.update_layout(
        showlegend=True,
        title_text="New York Times vs New York Post Stacked Sentiment Analysis Comparison from 2002-2022",
        title_x=0.5,
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'),
        xaxis2_type='category',
        yaxis2=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'),
        paper_bgcolor='rgb(201, 204, 209)')

    # adds a vertical line delineating an important event relating to the current topic
    for k, v in topic_dict.items():
        if topic == k:
            for i in range(len(v)):
                stackfig.add_vline(x=v[i]['x_vline'], line_width=2, line_dash="dash", line_color="black", layer='above')
                stackfig.add_annotation(x=v[i]['x_annotation'], text=v[i]['text'], showarrow=False, textangle=-90)
                stackfig.add_annotation(x=v[i]['x_annotation'], xref='x2', text=v[i]['text'], showarrow=False,
                                        textangle=-90)
    # return fig
    return stackfig


def read_descriptions(file_name):
    """
    reads in text file descriptions as strings to use in dashboard
    """
    with open(file_name, 'r') as file:
        description = file.read()

    return description


def main():
    # establish title and topic
    topic = 'abortion'
    title = f'Comparison of Media Coverage on Prevalent Social Issues: {" ".join([x.capitalize() for x in topic.split()])}'
    topic_graph_title = f'Sub Topics Generated Through Topic Modelling for ' \
                        f'{" ".join([x.capitalize() for x in topic.split()])}'

    # title for wordcloud graph
    word_title = f'Word Cloud Comparison of New York Times and New York Post for Words Used in ' \
                 f'{" ".join([x.capitalize() for x in topic.split()])} Articles (2002-2022)'

    source1_imagefile = 'New-York-Times-logo.png'
    source1_encoded = base64.b64encode(open(source1_imagefile, 'rb').read()).decode('ascii')

    source2_imagefile = 'nyp_logo.png'
    source2_encoded = base64.b64encode(open(source2_imagefile, 'rb').read()).decode('ascii')

    # load texts for topic
    source1_texts, source2_texts, df = load_topic(2002, 2022, topic, negs)

    # topic modeling functions
    words_df = topic_mdl.word_normalization(source1_texts, source2_texts)
    words_df, words_dict = topic_mdl.topic_modeling(words_df)

    # make visualization figures
    s = sank(8, 2013, topic, negs)
    p = parallel(2002, 2002, topic, negs)
    w1, w2 = wordcloud(source1_texts, source2_texts)
    st = stacked(df, topic, topics, negs)

    my_path = os.path.abspath(__file__)
    my_path = my_path.replace('/FinalProjectDashboardApp.py', '')
    file_path = my_path + f'/{topic}_network.png'

    if exists(file_path) == True:
        node_path = file_path
    else:
        node_path = topic_mdl.topic_modeling_graph(topic, words_df, words_dict)

    nodes_encoded = base64.b64encode(open(node_path, 'rb').read()).decode('ascii')


    # range of years, making the steps for the marks
    min_year = 2002
    max_year = 2022
    poss_years = list(range(min_year, max_year + 1))
    marks = {}
    for num in range(min_year, max_year + 1, 1):
        marks[num] = str(num)

    # read in description files for dashboard
    sank_desc = read_descriptions('sankey_desc.txt')
    wc_desc = read_descriptions('wc_desc.txt')
    stacked_desc = read_descriptions('stack_desc.txt')
    parallel_desc = read_descriptions('parallel_desc.txt')
    tm_desc = read_descriptions('tm_desc.txt')


    # make app

    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    app.layout = html.Div([

        # top line of Dash
        dbc.Row([
            # first logo
            dbc.Col(
                html.Img(src='data:image/png;base64,{}'.format(source1_encoded),
                         style={'height': '100%', 'width': '100%'}),
                width={'size': 2}),

            # set title
            dbc.Col(id='title_words', children=[
                html.H1(title, style={'text-align': 'center'})], width={'size': 6}),

            # second logo
            dbc.Col(
                html.Img(src='data:image/png;base64,{}'.format(source2_encoded),
                         style={'height': '100%', 'width': '100%'}),
                width={'size': 2})], justify="around"),

        dbc.Row([
            dbc.Col(
                # adding description for topic dropdown
                html.P("Select a topic:"),
                width={'size': 2, 'offset': 5}
            )
        ]),

        dbc.Row([
            dbc.Col(
                # topic selector dropdown
                dcc.Dropdown(id='topic_selector',
                             options=['abortion', 'gay marriage', 'marijuana', 'immigration', 'police brutality'],
                             value='abortion',
                             className='mb-4',
                             style={'text-align': 'center'}
                             ),
                width={'size': 2})], justify='center'),

        # tab containing word count visualizations
        dcc.Tabs([
            # tab containing network graph
            dcc.Tab(label='Media Topic Modeling', children=[
                dbc.Row([
                    dbc.Col(
                        html.P('', style={'font-size': '60px'})
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        # topic modeling description
                        html.P(tm_desc, className='mb-4', style={'text-align': 'left', 'font-size': '17px',
                                                                 'color': 'rgb(51, 56, 82)'
                                                                 }),
                        width={'size': 8}
                    )
                ], justify='center', align='end'),

                dbc.Row([
                    dbc.Col(
                        html.P('NYT vs NYP Node Graph of Sub-Topic Assignments (2002-2022)',
                               style={'font-size': '20px', 'text-align': 'center', 'color': 'rgb(51, 56, 82)'}
                               )
                    )
                ], justify='center'),


                dbc.Row([
                    dbc.Col(
                        # topic modeling graph
                        html.Img(id='node_path', src='data:image/png;base64,{}'.format(nodes_encoded)),
                        width={'size': 8, 'offset': 2}
                    )
                ], justify='center')
            ]),

            # tab containing sentiment visualizations
            dcc.Tab(label='Sentiment Analyses', children=[
                dbc.Row([
                    dbc.Col(
                        html.P('', style={'font-size': '60px'})
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        # description for stack plot
                        html.P(stacked_desc, className='mb-4', style={'text-align': 'left', 'font-size': '17px',
                                                                      'color': 'rgb(51, 56, 82)'}),
                        width={'size': 8}
                    )
                ], justify='center', align='center'),
                dbc.Row([
                    dbc.Col(
                        # stacked visualization
                        dcc.Graph(id='stacked', figure=st),
                        width={'size': 8, 'offset': 2}
                    )]),

                dbc.Row([
                    dbc.Col(
                        # description for parallel plot
                        html.P(parallel_desc, className='mb-4', style={'text-align': 'left', 'font-size': '17px',
                                                                       'color': 'rgb(51, 56, 82)'}),
                        width={'size': 8}
                    )
                ], justify='center', align='end'),

                dbc.Row([
                    dbc.Col(
                        # title for parallel plot
                        html.P('Parallel Coordinate Comparison of NYT & NYP Article Sentiments for a Given Year',
                               style={'font-size': '20px', 'text-align': 'center', 'color': 'rgb(51, 56, 82)'}
                               )
                    )
                ], justify='center'),

                dbc.Row([
                    dbc.Col(
                        # parallel coordinate visualization
                        dcc.Graph(id='parallel_fig', figure=p),
                        width={'size': 8, 'offset': 2})
                ]),

                dbc.Row([
                    dbc.Col(
                        # description for year slider
                        html.P('Select a Year:', className='mb-4'),
                        width={'size': 2, 'offset': 1})
                ]),

                dbc.Row([
                    dbc.Col(
                        # year selector for parallel coordinate visualization
                        dcc.Slider(id='parallel_year', min=min_year, max=max_year, step=None, marks=marks, value=2002,
                                   className='mb-4'),
                        width={'size': 10}
                    )
                ], justify='center')
            ]),

            # tab containing word analysis
            dcc.Tab(label='Word Usage Comparisons', children=[
                dbc.Row([
                    dbc.Col(
                        # sankey visualization
                        dcc.Graph(id='sankey_fig', figure=s),
                        width={'size': 9}
                    ),
                    dbc.Col(
                        # adding sankey graph description
                        html.P(sank_desc, style={'text-align': 'left', 'font-size': '17px', 'color': 'rgb(51, 56, 82)'
                                                 }, className='mb-5'),
                        width={'size': 2}
                    )
                ], align='center'),
                dbc.Row([
                    dbc.Col(
                        # adding text for dropdown
                        html.P("Select a Year:"),
                        width={'size': 2, 'offset': 2}
                    )
                ]),

                dbc.Row([
                    dbc.Col(
                        # year selector for sankey visualization
                        dcc.Dropdown(id='sankey_year', options=poss_years, value=2013, className='mb-5'),
                        width={'size': 2, 'offset': 2}
                    )], align="center"),

                dbc.Row([
                    dbc.Col(
                        # word cloud title
                        html.P(word_title,
                               style={'font-size': '20px', 'text-align': 'center', 'color': 'rgb(51, 56, 82)'})
                    )
                ], justify='center', align="end"),

                dbc.Row([
                    dbc.Col(
                        # wordcloud for source 1
                        dcc.Graph(id='wordcloud_source1', figure=w1), width={'size': 5}),

                    dbc.Col(
                        # wordcloud for source 2
                        dcc.Graph(id='wordcloud_source2', figure=w2), width={'size': 5}),
                ], justify="around", align="start"),

                dbc.Row([
                    dbc.Col(
                        # wordcloud graph description
                        html.P(wc_desc, className='mb-4', style={'text-align': 'left', 'font-size': '17px',
                                                                 'color': 'rgb(51, 56, 82)'}),
                        width={'offset': 1}
                    )
                ])
            ]),
        ], colors={
            "border": "rgb(250,251,255)",
            "primary": "rgb(153,184,255)",
            "background": "rgb(127, 152, 212)"}
        )], style={'background-color': 'rgb(201, 204, 209)'})

    # callbacks
    @app.callback(
        Output(component_id='title_words', component_property='children'),
        Output(component_id='wordcloud_source1', component_property='figure'),
        Output(component_id='wordcloud_source2', component_property='figure'),
        Output(component_id='stacked', component_property='figure'),
        Output(component_id='node_path', component_property='src'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_topic(topic_selector):
        # update title and wordcloud, stacked, and node graph visualizations with new topic
        title = f'Comparison of Media Coverage on Prevalent Social Issues: {" ".join([x.capitalize() for x in topic_selector.split()])}'
        header = html.H1(title, style={'text-align': 'center'})
        source1_texts, source2_texts, df = load_topic(min_year, max_year, topic_selector, negs)
        w1, w2 = wordcloud(source1_texts, source2_texts)
        st = stacked(df, topic_selector, topics, negs)

        words_df = topic_mdl.word_normalization(source1_texts, source2_texts)
        words_df, words_dict = topic_mdl.topic_modeling(words_df)

        # update node graph
        my_path = os.path.abspath(__file__)
        my_path = my_path.replace('/FinalProjectDashboardApp.py', '')
        file_path = my_path + f'/{topic_selector}_network.png'

        if exists(file_path) == True:
            node_path = file_path
        else:
            node_path = topic_mdl.topic_modeling_graph(topic_selector, words_df, words_dict)
        nodes_encoded = base64.b64encode(open(node_path, 'rb').read()).decode('ascii')
        src = 'data:image/png;base64,{}'.format(nodes_encoded)

        return header, w1, w2, st, src

    @app.callback(
        Output(component_id='sankey_fig', component_property='figure'),
        Input(component_id='sankey_year', component_property='value'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_sankey(sankey_year, topic_selector):
        # updates sankey based on new topic and year
        s = sank(8, sankey_year, topic_selector, negs)
        return s

    @app.callback(
        Output(component_id='parallel_fig', component_property='figure'),
        Input(component_id='parallel_year', component_property='value'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_parallel(parallel_year, topic_selector):
        # updates parallel coordinate visualization based on new topic and year
        p = parallel(parallel_year, parallel_year, topic_selector, negs)
        return p

    # run server
    app.run_server(debug=True, port=8053)


if __name__ == "__main__":
    main()
