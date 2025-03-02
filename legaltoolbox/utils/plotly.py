from langchain_core.documents import Document
import pandas as pd
import numpy as np

MAX_CHARACTERS_PER_LINE = 50

def to_rgba(colors: np.array) -> list[str]:
    if np.max(colors) > 1:
        return [f'rgb({c[0]:0f},{c[1]:0f},{c[2]:0f},{c[3]:0f})' for c in colors]
    else:
        return [f'rgb({255*c[0]:0f},{255*c[1]:0f},{255*c[2]:0f},{255*c[3]:0f})' for c in colors]
    
def format_text_for_plotly(text):

    if type(text) == Document:
        text = text.page_content

    if type(text) in [list, pd.Series, np.ndarray]:
        return [format_text_for_plotly(t) for t in text]
    
    if "<br>" in text:
        return "<br>".join([format_text_for_plotly(t) for t in text.split("<br>")])
    
    if "\n" in text:
        return "<br><br>".join([format_text_for_plotly(t) for t in text.split("\n")])

    text = text.split(" ")
    out = []
    for word in text:
        if len(out) == 0 or len(out[-1]) + len(word) > MAX_CHARACTERS_PER_LINE:
            out.append(word)
        else:
            out[-1] += " " + word
    return "<br>".join(out)