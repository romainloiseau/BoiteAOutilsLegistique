import streamlit as st

import hydra
import streamlit as st
import os.path as osp
import logging
import time
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate

from legaltoolbox import Application

logger = logging.getLogger(__name__)

def vizu_banner(interface, text):
    cols = st.columns(len(interface))
    for i, ((button_name, button), col) in enumerate(zip(interface.items(), cols)):
        with col:
            if "download" in button_name:
                button.file_name = text.name + button.file_name
                with open(getattr(text, button_name)(), "rb") as f:
                    instantiate(button, key=f"banner_{button_name}_{text.name}", data=f)
            else:
                if instantiate(button, key=f"banner_{button_name}_{text.name}"):
                    getattr(text, button_name)()
def reset_app():
    for key in ["appli", "logging_path", "logger", "llm", "embedding_model"]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()

def start():
    """
    Decorator function that initializes the Streamlit app with the given configuration name.

    Returns:
        function: The decorated function that represents the Streamlit app.
    """
    def Inner(page):
        def wrapper(*args, **kwargs):
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            hydra.initialize(config_path="../../configs", version_base="1.1")
            cfg = hydra.compose(config_name="defaults")
            hydra.core.global_hydra.GlobalHydra.instance().clear()

            if hasattr(cfg.perso, "langsmith_api_key") and cfg.perso.langsmith_api_key is not None:
                os.environ["LANGCHAIN_PROJECT"] = f"LegalToolbox"
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = cfg.perso.langsmith_api_key
                os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

            st.set_page_config(
                page_title=cfg.interface.page_title,
                page_icon=cfg.interface.page_icon,
                layout="wide"
            )
                
            if not osp.exists(cfg.root_path):
                os.makedirs(cfg.root_path)

            if "logging_path" in st.session_state:
                LOGGING_PATH = st.session_state.logging_path
            else:
                LOGGING_PATH = None

            if (LOGGING_PATH is None):
                LOGGING_PATH = f"outputs/{time.strftime('%Y%m%d%H%M%S')}"
                st.session_state.logging_path = LOGGING_PATH
                
            log_cfg = False
            if not osp.exists(LOGGING_PATH):
                os.makedirs(LOGGING_PATH)
                log_cfg = True

            LOGGING_CONFIG = {
                    'version': 1,
                    'disable_existing_loggers': False,
                    'formatters': {
                        'colored': {
                            '()': 'colorlog.ColoredFormatter',
                            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                        },
                    },
                    'handlers': {
                        'console_handler': {
                            'level': 'INFO',
                            'formatter': 'colored',
                            'class': 'logging.StreamHandler',
                            'stream': 'ext://sys.stdout',  # Default is stderr
                        },
                        'file_handler': {
                            'level': 'INFO',
                            'formatter': 'colored',
                            'class': 'logging.FileHandler',
                            'filename': f"{LOGGING_PATH}/out.log",
                        },
                        },
                    'loggers': {
                        '': {
                            'handlers': ['console_handler', 'file_handler'],
                            'level': 'INFO',
                            'propagate': True,
                        },
                    }
                }

            logging.config.dictConfig(LOGGING_CONFIG)

            st.session_state["logger"] = logging.getLogger(st.__name__)
                
            with st.sidebar:
                instantiate(cfg.interface.sidebar.welcome)
                
                with instantiate(cfg.interface.sidebar.pro_expander):
                    with st.form("llm_choice"):
                        llm = instantiate(cfg.interface.sidebar.llm_select, _recursive_=False)
                        if instantiate(cfg.interface.sidebar.llm_submit):
                            st.session_state["llm"] = cfg.models.llms[llm]
                            st.rerun()

                    with st.form("embedding_model_choice"):
                        embedding_model = instantiate(cfg.interface.sidebar.embedding_model_select, _recursive_=False)
                        if instantiate(cfg.interface.sidebar.embedding_model_submit):
                            st.session_state["embedding_model"] = cfg.models.embedding_models[embedding_model]
                            st.rerun()
                    
                instantiate(cfg.interface.sidebar.credits)
                st.divider()

                if hasattr(st.session_state, "appli"):
                    if st.session_state.appli.fr_text is not None:
                        st.markdown(f"## {st.session_state.appli.fr_text.name}")
                        button_name = list(cfg.interface.fr_text.row_buttons.keys())[0]
                        button = cfg.interface.fr_text.row_buttons[button_name]
                        if instantiate(button, key=f"sidebar_{button_name}_{st.session_state.appli.fr_text.name}"):
                            getattr(st.session_state.appli.fr_text, button_name)()

                    if st.session_state.appli.eu_text is not None:
                        st.markdown(f"## {st.session_state.appli.eu_text.name}")
                        button_name = list(cfg.interface.eu_text.row_buttons.keys())[0]
                        button = cfg.interface.eu_text.row_buttons[button_name]
                        if instantiate(button, key=f"sidebar_{button_name}_{st.session_state.appli.eu_text.name}"):
                            getattr(st.session_state.appli.eu_text, button_name)()

                    if st.session_state.appli.eu_text is not None or st.session_state.appli.fr_text is not None:
                        st.divider()                

                if instantiate(cfg.interface.sidebar.reset):
                    reset_app()

            if not hasattr(st.session_state, "llm"):
                st.session_state["llm"] = cfg.models.llms[list(cfg.models.llms.keys())[0]]

            if not hasattr(st.session_state, "embedding_model"):
                st.session_state["embedding_model"] = cfg.models.embedding_models[list(cfg.models.embedding_models.keys())[0]]
                    
            cfg.llm = st.session_state["llm"]
            cfg.embedding_model = st.session_state["embedding_model"]

            if not hasattr(st.session_state, "appli"):
                st.session_state.appli = Application(cfg)
                
            if log_cfg:
                logger.info("Starting with config:\n" + OmegaConf.to_yaml(cfg))
                
            page(cfg)

        return wrapper
    return Inner
