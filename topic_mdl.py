import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from nltk.corpus import stopwords
import plotly.graph_objects as go
import re as re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from PIL import Image

import os
import base64



def word_normalization(nyt_texts, nyp_texts):

    stops = list(set(stopwords.words('english')))
    df = pd.DataFrame(columns=["text", "news_outlet"])

    for year in range(2002, 2023):
        nyt_string = nyt_texts[year]
        nyp_string = nyp_texts[year]

        # remove numbers from strings
        nyt_string = re.sub(r'\d+', '', nyt_string)
        nyp_string = re.sub(r'\d+', '', nyp_string)

        # remove other punctuation
        nyt_string = re.sub(r'[^\w\s]','', nyt_string)
        nyp_string = re.sub(r'[^\w\s]','', nyp_string)

        # split strings
        nyt = nyt_string.split()
        nyp = nyp_string.split()

        # remove stop words
        for word in stops:
            nyt = [value for value in nyt if value != word]
            nyp = [value for value in nyp if value != word]


        # rejoin to string
        nyt = ' '.join(nyt)
        nyp = ' '.join(nyp)

        df = df.append({"text": nyt, "news_outlet": "nyt"}, ignore_index=True)
        df = df.append({"text": nyp, "news_outlet": "nyp"}, ignore_index=True)

    return df


def topic_modeling(df):
    """
    creates df where subtopics are classified and words are assigned to subtopics
    :param df: dataframe object,

    """

    # initialize count vectorizer
    cv = CountVectorizer(max_df=.95, min_df=2)

    # create document term matrix
    dtm = cv.fit_transform(df["text"])

    # build lda model and fit to document term matrix
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    wrds_dict = {}

    # retrieve words of each topic, save in dict
    for i, topic in enumerate(lda.components_):
        words = [cv.get_feature_names()[index] for index in topic.argsort()[-5:]]
        wrds_dict[i] = words

    topic_results = lda.transform(dtm)
    df["topic"] = topic_results.argmax(axis=1)

    return df, wrds_dict


def topic_modeling_graph(title, df, wrds_dict):


    nyt = df[df["news_outlet"] == "nyt"]
    nyt = nyt.groupby(["topic"]).size().reset_index(name="weight")
    nyt["journal"] = "NYTimes"

    nyp = df[df["news_outlet"] == "nyp"]
    nyp = nyp.groupby(["topic"]).size().reset_index(name="weight")
    nyp["journal"] = "NYPost"

    topic_edge = pd.concat([nyp, nyt], axis=0)

    topic_edge["top words"] = topic_edge["topic"].map(wrds_dict)
    topic_edge = topic_edge.replace({"topic": {0: "Topic 1", 1: "Topic 2", 2: "Topic 3", 3: "Topic 4", 4: "Topic 5"}})
    G = nx.from_pandas_edgelist(topic_edge, "journal", "topic")

    # setting size attr based on frequency/relevance
    attr = topic_edge.copy().groupby("topic").sum().reset_index()
    attr["weight"] = [300+(weight * 25) if weight >= 10 else 300+(weight * 45) for weight in attr["weight"]]
    node_size = attr.set_index("topic")["weight"].to_dict()
    node_size["NYTimes"] = 1000
    node_size["NYPost"] = 1000

    wrd_edge = topic_edge[["topic", "top words"]][:5]
    wrd_edge = wrd_edge.explode("top words").reset_index(drop=True)

    H = nx.from_pandas_edgelist(wrd_edge, "topic", "top words")

    for node in list(H.nodes):
        if "Topic" not in node:
            n = len(list(H.neighbors(node)))
            node_size[node] = 300+(40*n)

    # setting color gradient
    node_clr = {}
    for k,v in node_size.items():
        if v == 1000:
            node_clr[k] = .7
        elif 550 <= v < 1000:
            node_clr[k] = .55
        else:
            node_clr[k] = .4

    N = nx.compose(G, H)

    nx.set_node_attributes(N, node_size, "size")
    nx.set_node_attributes(N, node_clr, "color")

    plt.Figure(figsize=(18,50))

    nx.draw_networkx(N, pos=nx.spring_layout(N, scale=5),
                     node_size=[N.nodes[n]["size"] for n in N.nodes],
                     node_color=[N.nodes[n]["color"] for n in N.nodes],
                     cmap=plt.cm.Blues, vmin=0, vmax=1,
                     font_weight='bold',
                     with_labels=True)


    # gets absolute path
    my_path = os.path.abspath(__file__)
    my_path = my_path.replace('/topic_mdl.py', '')
    file_path = my_path + f'/{title}_network.png'

    plt.savefig(file_path, transparent=True)
    #networkx_encoded = base64.b64encode(open(file_path, 'rb').read()).decode('ascii')

    # image = Image.open('network.png')

    '''
    fig = go.Figure()
    fig.add_trace(go.Image(ntviz))
    fig.update_layout(height=800, width=800,
                       title_text='Network Graph for Topic Modeling– Most Relevant Topics and Words')
                       
    '''

    return file_path