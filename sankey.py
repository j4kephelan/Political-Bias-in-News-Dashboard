import plotly.graph_objects as go
import pandas as pd


def _code_mapping(df, src, targ):
    """ Maps labels / strings in src and target
    and converts them to integers 0,1,2,3... """

    # Extract distinct labels
    # labels = sorted(list(set(list(df[src]) + list(df[targ]))))
    labels = list(set(list(df[src]) + list(df[targ])))

    # define integer codes
    codes = list(range(len(labels)))

    # pair labels with list
    lc_map = dict(zip(labels, codes))

    # in df, substitute codes for labels
    df = df.replace({src: lc_map, targ: lc_map})


    return df, labels


def make_sankey(df, src, targ, vals, year, topic, **kwargs):
    """Generate the sankey diagram """

    df, labels = _code_mapping(df, src, targ)

    if vals:
        values = df[vals]
    else:
        values = [1] * len(df)

    pad = kwargs.get('pad', 50)
    thickness = kwargs.get('thickness', 30)
    line_color = kwargs.get('line_color', 'black')
    line_width = kwargs.get('line_width', 1)

    first = list(df["src"])[0]
    for i in range(len(list(df["src"]))):
        if list(df["src"])[i] != first:
            next = list(df["src"])[i]

    """
    + df[src][next:next+10]
    + df[targ][next:next+10]
    + values[next:next+10]
    +labels[next:next+10]
    """


    link = {'source': df[src], 'target': df[targ] ,
            'value': values }
    node = {'label': labels, 'pad': pad, 'thickness': thickness,
            'line': {'color': line_color, 'width': line_width}}

    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)

    fig.update_layout(
        title=f"Most Used Nouns for NYT vs. NYP in {topic.capitalize()} Articles ({year})",
        title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

    return fig
    #fig.write_image(png)
