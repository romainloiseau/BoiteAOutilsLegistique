import copy
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import instantiate
import requests

import os.path as osp

import legaltoolbox
import legaltoolbox.tools as tools
from legaltoolbox.utils.load_text import load_fr_text

class Application:

    def __init__(self, *args, **kwargs):
        self.cfg = copy.deepcopy(kwargs)

        self.fr_text = None
        self.eu_text = None
        self.out_analyse_impact = None

    def load_fr_text(self, cfg, existing_codes):

        cfg_fr_text = copy.deepcopy(cfg.data.fr_text)
        cfg_fr_text.code_names = existing_codes

        self.fr_text: legaltoolbox.data.LegifranceData = instantiate(cfg_fr_text, _recursive_=False)

    def load_eu_text(self, cfg, project_name, eu_text = None, start_formula = None, end_formula = None):

        cfg_eu_text = copy.deepcopy(cfg.data.eu_text)
        cfg_eu_text.folder = project_name

        if start_formula is not None:
            cfg_eu_text.start_formula = start_formula
        if end_formula is not None:
            cfg_eu_text.end_formula = end_formula

        self.eu_text: legaltoolbox.data.EUData = instantiate(cfg_eu_text, eu_text, _recursive_=False)

        if not self.eu_text.success:
            instantiate(cfg.interface.eu_text.new.invalid_text)
            self.eu_text = self.eu_text.delete_folder_and_return_none()
            st.stop()

    def load_eu_text_from_id(self, cfg, id, project_name, start_formula, end_formula):

        request_link = requests.get(cfg.data.eu_text.url + id)
        
        if request_link.status_code != 200:
            instantiate(cfg.interface.eu_text.new.from_id.invalid_url)
            st.stop()

        eu_text = request_link.text

        return self.load_eu_text(cfg, project_name, eu_text, start_formula, end_formula)

    def load_eu_text_from_file(self, cfg, file, project_name, start_formula, end_formula):
        return self.load_eu_text(cfg, project_name, file, start_formula, end_formula)

    def transposition_table(self, cfg):

        instantiate(cfg.interface.application.transposition_table.warning_computation_cost)

        if instantiate(cfg.interface.application.transposition_table.confirm_run):
            docx_save_path = osp.join(st.session_state.logging_path, "tableau_de_transpo.docx")

            if self.out_analyse_impact is None:
                self.impact_analysis_of_eu_on_fr(cfg, output_streamlit = False)

            if not osp.exists(docx_save_path):
                eu_texts = self.eu_text.get_texts(need_transposition=True).to_list()

                tools.run_transposition_table(cfg, eu_texts, self.eu_text, self.out_analyse_impact, docx_save_path)

            with open(docx_save_path, "rb") as f:
                instantiate(cfg.interface.application.transposition_table.download_docx, data=f)

    def impact_analysis_of_eu_on_fr(self, cfg, output_streamlit = True):

        if self.fr_text is None:
            load_fr_text(cfg)

        html_save_path = osp.join(st.session_state.logging_path, "impact_analysis.html")

        if not osp.exists(html_save_path):
            fr_texts = self.fr_text.get_texts(focus=None)
            fr_embeddings = self.fr_text.get_embeddings(focus=None)

            self.eu_text.is_transposition_needed()
            eu_embeddings = self.eu_text.get_embeddings(ids = self.eu_text.CONTENT_ID)

            fig, html, self.out_analyse_impact = tools.run_analyse_impact(cfg, fr_embeddings, fr_texts, eu_embeddings, html_save_path)
            self.out_analyse_impact["fig"] = fig
            self.out_analyse_impact["html"] = html

        if output_streamlit:
            with open(html_save_path, "rb") as f:
                instantiate(cfg.interface.application.impact_analysis_of_eu_on_fr.download_html, data=f)

            st.plotly_chart(self.out_analyse_impact["fig"])
            st.html(self.out_analyse_impact["html"])

    def projection_fr(self, cfg):

        """
        columns = st.columns(2)

        with columns[0]:
            type = instantiate(cfg.interface.fr_text.type_choice)
        with columns[1]:
            if type is not None:
                value = st.selectbox(type.capitalize(), np.unique(self.fr_text.pandas_codes[type.lower()]))
        """

        texts = [f"<b>{row['uid']}</b><br>{row['content']}" for index, row in self.fr_text.pandas_codes.iterrows()]

        cmap = plt.get_cmap("tab20c")
        unique_parts = np.unique(self.fr_text.pandas_codes["partie"])
        i_parts = {part: i for i, part in enumerate(unique_parts)}

        unique_books = np.unique(self.fr_text.pandas_codes["livre"])
        i_books = {book: i for i, book in enumerate(unique_books)}

        colors = cmap((self.fr_text.pandas_codes["partie"].map(i_parts) + 4*self.fr_text.pandas_codes["livre"].map(i_books)) % 20)

        embeddings = self.fr_text.embeddings
        
        st.plotly_chart(tools.run_project(cfg, [embeddings], [texts], [colors], [self.fr_text.name]))

    def projection_eu(self, cfg):
        texts = [self.eu_text.get_texts(ids = self.eu_text.CONTENT_ID)]
        embeddings = [self.eu_text.get_embeddings(ids = self.eu_text.CONTENT_ID)]

        names = [self.eu_text.name]
        
        colors = [plt.get_cmap("autumn")(np.arange(len(texts[0])) / len(texts[0]))]

        if self.fr_text is not None:
            texts.append([f"<b>{row['uid']}</b><br>{row['content']}" for index, row in self.fr_text.pandas_codes.iterrows()])
            embeddings.append(self.fr_text.embeddings)
            colors.append(plt.get_cmap("winter")(np.arange(len(self.fr_text.pandas_codes)) / len(self.fr_text.pandas_codes)))
            names.append(self.fr_text.name)

        st.plotly_chart(tools.run_project(cfg, embeddings, texts, colors, names))

    def summary_eu(self, cfg):
        instantiate(cfg.interface.application.summary_eu.warning_computation_cost)

        if instantiate(cfg.interface.application.summary_eu.confirm_run):
            texts = self.eu_text.get_texts(ids = self.eu_text.ARTICLE_TITLE_ID)
            embeddings = self.eu_text.get_embeddings(ids = self.eu_text.ARTICLE_TITLE_ID)
            full_texts = self.eu_text.articles

            doc_save_path = osp.join(st.session_state.logging_path, "clustering.docx")
            html, fig = tools.run_cluster_and_project(cfg, embeddings, texts, full_texts, doc_save_path)

            with open(doc_save_path, "rb") as f:
                instantiate(cfg.interface.application.summary_eu.download_docx, data=f)
                
            st.plotly_chart(fig)
            st.html(html)