import streamlit as st
from hydra.utils import instantiate

from legaltoolbox.utils import streamlit as stp
from legaltoolbox.utils.load_text import load_fr_text

@stp.start()
def page(cfg, **kwargs):
    instantiate(cfg.interface.fr_text.intro)

    if st.session_state.appli.fr_text is None:
        load_fr_text(cfg)

    stp.vizu_banner(cfg.interface.fr_text.row_buttons, st.session_state.appli.fr_text)

    tool_choice = instantiate(cfg.interface.fr_text.tool_choice)

    if tool_choice is not None:
        with instantiate(cfg.interface.fr_text.spinner_tool):
            getattr(st.session_state.appli, cfg.interface.fr_text.tool_choice.options[tool_choice])(cfg)
    
if __name__ == "__main__":
    page()