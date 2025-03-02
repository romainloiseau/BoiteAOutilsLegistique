import streamlit as st
import plotly.graph_objects as go
from hydra.utils import instantiate
import numpy as np

from legaltoolbox.utils import plotly as myplotly

@st.cache_data(show_spinner = False)
def run_project(_cfg, embeddings: list[np.array], texts: list[list], colors: list[np.array], names = list[str]) -> go.Figure:

    projection = instantiate(_cfg["tsne"]).fit_transform(np.concatenate(embeddings))

    data = []
    start, end = 0, 0
    for e, t, c, n in zip(embeddings, texts, colors, names):
        start, end = end, end + len(e)

        data.append(go.Scatter(
            x=projection[start:end, 0],
            y=projection[start:end, 1],
            mode='markers',
            marker_color=myplotly.to_rgba(c),
            hovertext=myplotly.format_text_for_plotly(t),
            hoverinfo="text",
            name=n
        ))

    fig = go.Figure(data=data)

    fig.update_layout(title='Projection', xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_layout({ax:{"visible":False, "matches":None} for ax in fig.to_dict()["layout"] if "axis" in ax})

    return fig