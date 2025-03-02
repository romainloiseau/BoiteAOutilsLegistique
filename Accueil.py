import os
import copy
from hydra.utils import instantiate

from legaltoolbox.utils import streamlit as stp

@stp.start()
def page(cfg, **kwargs):
    instantiate(cfg.interface.hello.intro)

    with instantiate(cfg.interface.hello.fr_text.expander):
        for path in os.listdir(cfg.data.fr_text.path):
            cfg_fr_text = copy.deepcopy(cfg.data.fr_text)
            cfg_fr_text.code_names = path.split(" - ")
            instantiate(cfg_fr_text, _recursive_=False).show_in_streamlit(cfg.interface.hello.fr_text.show_content, key=f"show_content_fr_text_{path}")

    with instantiate(cfg.interface.hello.eu_text.expander):
        for path in os.listdir(cfg.data.eu_text.path):
            cfg_eu_text = copy.deepcopy(cfg.data.eu_text)
            cfg_eu_text.folder = path
            instantiate(cfg_eu_text, _recursive_=False).show_in_streamlit(cfg.interface.hello.eu_text.show_content, key=f"show_content_eu_text_{path}")

if __name__ == "__main__":
    page()