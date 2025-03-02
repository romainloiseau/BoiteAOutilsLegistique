import streamlit as st
from hydra.utils import instantiate
import os
import os.path as osp
import legaltoolbox.utils.load_text as lt

def is_valid_folder_name(name):
    if name is None:
        return False
    if name == "" or " " in name:
        return False
    return True

def import_new_eu_text_from_id(cfg):
     with st.form("eur_lex_new_text_form"):
        project_name = instantiate(cfg.interface.eu_text.new.project_name)
        id = instantiate(cfg.interface.eu_text.new.from_id.id)

        start_formula = instantiate(cfg.interface.eu_text.new.start_formula)
        end_formula = instantiate(cfg.interface.eu_text.new.end_formula)

        if instantiate(cfg.interface.eu_text.new.submit):
            if not is_valid_folder_name(project_name):
                instantiate(cfg.interface.eu_text.new.invalid_project_name)
                st.stop()
            elif project_name in os.listdir(cfg.data.eu_text.path):
                instantiate(cfg.interface.eu_text.new.existing_project_name)
            else:
                with instantiate(cfg.interface.eu_text.new.spinner_analyzing):
                    st.session_state.appli.load_eu_text_from_id(
                        cfg, id, project_name, start_formula, end_formula)
                st.rerun()
        else:
            st.stop()

def import_new_eu_text_from_upload(cfg):
    with st.form("eur_lex_new_text_form_from_upload"):
        project_name = instantiate(cfg.interface.eu_text.new.project_name)
        file = instantiate(cfg.interface.eu_text.new.from_file.file)

        start_formula = instantiate(cfg.interface.eu_text.new.start_formula)
        end_formula = instantiate(cfg.interface.eu_text.new.end_formula)

        if instantiate(cfg.interface.eu_text.new.submit):
            if not is_valid_folder_name(project_name):
                instantiate(cfg.interface.eu_text.new.invalid_project_name)
                st.stop()
            elif project_name in os.listdir(cfg.data.eu_text.path):
                instantiate(cfg.interface.eu_text.new.existing_project_name)
            else:
                with instantiate(cfg.interface.eu_text.new.spinner_analyzing):
                    st.session_state.appli.load_eu_text_from_file(
                        cfg, file, project_name, start_formula, end_formula)
                st.rerun()
        else:
            st.stop()

def import_new_eu_text(cfg):

    if not osp.exists(cfg.data.eu_text.path):
        os.makedirs(cfg.data.eu_text.path)

    type = instantiate(cfg.interface.eu_text.new.type)

    if type:
        getattr(lt, cfg.interface.eu_text.new.type.options[type])(cfg)
    else:
        st.stop()

def load_existing_eu_text(cfg):

    cfg.interface.eu_text.existing.folder.options = {k: k for k in os.listdir(cfg.data.eu_text.path)}

    folder = instantiate(cfg.interface.eu_text.existing.folder)

    if folder is None:
        st.stop()
    else:
        st.session_state.appli.load_eu_text(cfg, folder)
        st.rerun()

def load_eu_text(cfg):
    which_text = instantiate(cfg.interface.eu_text.which_text)

    if which_text:
        getattr(lt, cfg.interface.eu_text.which_text.options[which_text])(cfg)
    else:
        st.stop()

def load_fr_text(cfg):
    with st.form("fr_text_form"):
        existing_codes = instantiate(cfg.interface.fr_text.code.select)

        if instantiate(cfg.interface.fr_text.code.select_submit):
            if len(existing_codes) == 0:
                instantiate(cfg.interface.fr_text.code.select_error)
                st.stop()
            else:
                with instantiate(cfg.interface.fr_text.code.wait_for_text_loading):
                    st.session_state.appli.load_fr_text(
                        cfg, existing_codes)
                st.rerun()
        else:
            st.stop()