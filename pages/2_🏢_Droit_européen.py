import streamlit as st
from hydra.utils import instantiate

from legaltoolbox.utils import streamlit as stp
from legaltoolbox.utils.load_text import load_eu_text

@stp.start()
def page(cfg, **kwargs):
    instantiate(cfg.interface.eu_text.intro)

    if st.session_state.appli.eu_text is None:
        load_eu_text(cfg)

    stp.vizu_banner(cfg.interface.eu_text.row_buttons, st.session_state.appli.eu_text)

    tool_choice = instantiate(cfg.interface.eu_text.tool_choice)

    if tool_choice is not None:
        with instantiate(cfg.interface.eu_text.spinner_tool):
            getattr(st.session_state.appli, cfg.interface.eu_text.tool_choice.options[tool_choice])(cfg)
    
if __name__ == "__main__":
    page()