import logging
import copy
logger = logging.getLogger(__name__)

import streamlit as st

import legaltoolbox

def show_recursive_dict(d, level=""):
    display_text = ""
    for k, v in d.items():
        if type(v) == dict:
            display_text += f"{level} {k}\n\n" + \
                show_recursive_dict(v, level=f"{level}#")
        else:
            display_text += f"**{k} :** "
            if v.startswith(" "):
                v = v[1:]
            display_text += v.replace('\n', '\n\n')
            display_text += "\n\n"
    return display_text

@st.dialog("Visualisation du texte", width="large")
def show_text_in_streamlit(text, title):

    display_text = f"# {title}\n\n"
    if type(text) == dict:
        for k, v in text.items():
            display_text += f"**{k} :** {v}\n\n"

    elif isinstance(text, legaltoolbox.data.eu_text.EUData) and hasattr(text, "content"):
        for i, row in text.content.iterrows():
            if row['id'] in text.HEAD_IDS:
                display_text += "**"
                display_text += row['text'].replace('\n', '\n\n')
                display_text += "**"
            else:
                display_text += row['text'].replace('\n', '\n\n')
            display_text += "\n\n"

    elif isinstance(text, legaltoolbox.data.legifrance.LegifranceData) and hasattr(text, "json_codes"):
        display_text += show_recursive_dict(text.json_codes, level="##")

    else:
        display_text += f"_L'outil de visualisation ne peut pas afficher ce type de texte ({type(text)})_"

    st.markdown(display_text)

class BaseData():
    EMBEDDINGS_FILE_NAME = "embeddings_{model_name}.npy"

    def __init__(self, *args, **kwargs):
        self.cfg = copy.deepcopy(kwargs)

    def get_texts(self, *args, **kwargs):
        raise NotImplementedError
    
    def show_text(self):
        show_text_in_streamlit(self, self.name)