import logging
import os.path as osp
from hydra.utils import to_absolute_path
from pylegifrance import LegiHandler, recherche_CODE
import json
import re
import pandas as pd
from .base import BaseData
import numpy as np
import hydra
import os
from langchain_core.documents import Document
import omegaconf
import streamlit as st

from hydra.utils import instantiate

logger = logging.getLogger(__name__)

def en_vigueur(article):
    return article["etat"] == "VIGUEUR"

def process_article(article):
    hrefs = re.findall(r'<a href=\'(.*?)\'.*?>', article)
    to_ids = []
    for href in hrefs:
        id = re.findall(r'&idArticle=(.*?)&', href)
        if len(id) > 0:
            to_ids.append(id[0])

    article = clean_article(article)

    return article, to_ids

def clean_article(text):
    text = text.replace("</p><p>", "\n")
    text = text.replace("<br/><p> <br/>", "\n")
    text = re.sub(r'<a.*?>(.*?)</a>', r'\1', text)
    text = re.sub(r'<font.*?>(.*?)</font>', r'\1', text)

    text = text.replace("<br>", "\n")
    text = text.replace("<br/>", "\n")
    text = text.replace("<br clear='none'/>", "\n")

    text = text.replace("<p>", "")
    text = text.replace("<p align='center'>", "")
    text = text.replace("<p align='left'>", "")
    text = text.replace("</p>", "")

    text = text.replace("<div>", "")
    text = text.replace("<div align='left'>", "")
    text = text.replace("</div>", "")

    text = text.replace("<sup>", "")
    text = text.replace("</sup>", "")

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\t+', ' ', text)
    return text

def process_title(title):
    # remove \n, \r, \t
    title = title.replace("\n", " ")
    title = title.replace("\r", " ")
    title = title.replace("\t", " ")

    # remove multiple spaces
    title = re.sub(r'\s+', ' ', title)

    if title.endswith("."):
        title = title[:-1]

    return title

class LegifranceData(BaseData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        code_names = self.cfg["code_names"][0]
        if len(self.cfg["code_names"]) > 1:
            code_names += " - " + \
                " - ".join([c.replace("Code ", "")
                           for c in self.cfg["code_names"][1:]])
            
        self.name = code_names

        self.path = to_absolute_path(osp.join(self.cfg["path"], code_names))
        if not osp.exists(self.path):
            os.makedirs(self.path)

        self.retrieve_codes(code_names=self.cfg["code_names"])
        self.retrieve_embeddings()

    def show_in_streamlit(self, show_content, key):
        st.markdown(f"## {self.name}")

        if instantiate(show_content, key=key):
            st.write(self.pandas_codes)

    def retrieve_codes(self, code_names: list | str | omegaconf.listconfig.ListConfig):
        if type(code_names) == str and code_name.starts_with("[") and code_name.ends_with("]"):
            code_names = code_names[1:-1].split(",")
        elif type(code_names) == omegaconf.listconfig.ListConfig:
            code_names = code_names._content

        assert type(
            code_names) == list, f"code_names should be a list of strings representing the names of the codes to retrieve. '{code_names}' of type {type(code_names)} is not a valid."

        filename = osp.join(self.path, f"codes.json")

        codes = {}
        pandas_codes = pd.DataFrame(columns=["code", "partie", "livre", "titre",
                                    "chapitre", "section", "sous-section", "article", "id", "content", "to_ids"])

        if not osp.exists(filename):
            logger.info(f"Setting up Legifrance API client")

            client = LegiHandler()
            client.set_api_keys(
                legifrance_api_key=self.cfg["legifrance_api_key"],
                legifrance_api_secret=self.cfg["legifrance_api_secret"],
            )

            for code_name in code_names:
                logger.info(f"Fetching '{code_name}' from Legifrance")
                code_text = recherche_CODE(str(code_name))[0]

                for article in code_text["articles"]:
                    if en_vigueur(article):
                        a, to_ids = process_article(article["content"])
                        codes[article["num"]] = a
                        pandas_codes.loc[len(pandas_codes)] = {
                            "code": code_name,
                            "partie": None,
                            "livre": None,
                            "titre": None,
                            "chapitre": None,
                            "section": None,
                            "sous-section": None,
                            "article": article["num"],
                            "id": article["id"],
                            "content": a, "to_ids": to_ids
                        }

                for partie in code_text["sections"]:
                    p = process_title(partie["title"])
                    codes[p] = {}

                    for article in partie["articles"]:
                        if en_vigueur(article):
                            a, to_ids = process_article(article["content"])
                            codes[p][article["num"]] = a
                            pandas_codes.loc[len(pandas_codes)] = {
                                "code": code_name,
                                "partie": p,
                                "livre": None,
                                "titre": None,
                                "chapitre": None,
                                "section": None,
                                "sous-section": None,
                                "article": article["num"],
                                "id": article["id"],
                                "content": a, "to_ids": to_ids
                            }

                    for livre in partie["sections"]:
                        l = process_title(livre["title"])
                        codes[p][l] = {}

                        for article in livre["articles"]:
                            if en_vigueur(article):
                                a, to_ids = process_article(article["content"])
                                codes[p][l][article["num"]] = a
                                pandas_codes.loc[len(pandas_codes)] = {
                                    "code": code_name,
                                    "partie": p,
                                    "livre": l,
                                    "titre": None,
                                    "chapitre": None,
                                    "section": None,
                                    "sous-section": None,
                                    "article": article["num"],
                                    "id": article["id"],
                                    "content": a, "to_ids": to_ids
                                }

                        for titre in livre["sections"]:
                            t = process_title(titre["title"])
                            codes[p][l][t] = {}

                            for article in titre["articles"]:
                                if en_vigueur(article):
                                    a, to_ids = process_article(
                                        article["content"])
                                    codes[p][l][t][article["num"]] = a
                                    pandas_codes.loc[len(pandas_codes)] = {
                                        "code": code_name,
                                        "partie": p,
                                        "livre": l,
                                        "titre": t,
                                        "chapitre": None,
                                        "section": None,
                                        "sous-section": None,
                                        "article": article["num"],
                                        "id": article["id"],
                                        "content": a, "to_ids": to_ids
                                    }

                            for chapitre in titre["sections"]:
                                c = process_title(chapitre["title"])
                                codes[p][l][t][c] = {}

                                for article in chapitre["articles"]:
                                    if en_vigueur(article):
                                        a, to_ids = process_article(
                                            article["content"])
                                        codes[p][l][t][c][article["num"]] = a
                                        pandas_codes.loc[len(pandas_codes)] = {
                                            "code": code_name,
                                            "partie": p,
                                            "livre": l,
                                            "titre": t,
                                            "chapitre": c,
                                            "section": None,
                                            "sous-section": None,
                                            "article": article["num"],
                                            "id": article["id"],
                                            "content": a, "to_ids": to_ids
                                        }

                                for section in chapitre["sections"]:
                                    s = process_title(section["title"])
                                    codes[p][l][t][c][s] = {}

                                    for article in section["articles"]:
                                        if en_vigueur(article):
                                            a, to_ids = process_article(
                                                article["content"])
                                            codes[p][l][t][c][s][article["num"]] = a
                                            pandas_codes.loc[len(pandas_codes)] = {
                                                "code": code_name,
                                                "partie": p,
                                                "livre": l,
                                                "titre": t,
                                                "chapitre": c,
                                                "section": s,
                                                "sous-section": None,
                                                "article": article["num"],
                                                "id": article["id"],
                                                "content": a, "to_ids": to_ids
                                            }

                                    for sous_section in section["sections"]:
                                        ss = process_title(
                                            sous_section["title"])
                                        codes[p][l][t][c][s][ss] = {}

                                        for article in sous_section["articles"]:
                                            if en_vigueur(article):
                                                a, to_ids = process_article(
                                                    article["content"])
                                                codes[p][l][t][c][s][ss][article["num"]] = a
                                                pandas_codes.loc[len(pandas_codes)] = {
                                                    "code": code_name,
                                                    "partie": p,
                                                    "livre": l,
                                                    "titre": t,
                                                    "chapitre": c,
                                                    "section": s,
                                                    "sous-section": ss,
                                                    "article": article["num"],
                                                    "id": article["id"],
                                                    "content": a, "to_ids": to_ids
                                                }

            uid = []
            for i, row in pandas_codes.iterrows():
                uid.append(f"Article {row['article']} du {row['code']}")
            pandas_codes["uid"] = uid

            with open(filename, "w") as f:
                f.write(json.dumps(codes, indent=4))

            pandas_codes.to_csv(osp.join(self.path, "code.csv"), index=False)

        else:
            logger.info(f"Code(s) already exists locally")
            with open(filename, "r") as f:
                codes = json.load(f)

            pandas_codes = pd.read_csv(osp.join(self.path, "code.csv"))

        logger.info(
            f"Retrieved {len(pandas_codes)} articles from Legifrance in {code_names}")

        self.json_codes = codes
        self.pandas_codes = pandas_codes

    def download_csv(self):
        return osp.join(self.path, "code.csv")
    
    def download_json(self):
        return osp.join(self.path, "codes.json")

    def get_texts(self, focus=None, full=False):
        if focus is not None:
            assert type(focus) == tuple
            assert focus[0] in ["code", "partie", "livre",
                                "titre", "chapitre", "section", "sous-section"]

        texts = []
        for i, row in self.pandas_codes.iterrows():
            text = ""

            if full:
                for col in ["code", "partie", "livre", "titre", "chapitre", "section", "sous-section"]:
                    if (row[col] is not None) and (row[col] is not np.nan):
                        text += row[col] + ", "
                text += row["article"] + ": "

            text += row["content"]

            if (focus is None) or (row[focus[0]] == focus[1]):
                texts.append(Document(page_content=text, metadata=dict(
                    uid=row["uid"], id=row["id"], to_ids=row["to_ids"])))

        return texts

    def get_embeddings(self, focus=None):
        if focus is not None:
            assert type(focus) == tuple
            assert focus[0] in ["code", "partie", "livre",
                                "titre", "chapitre", "section", "sous-section"]

        if focus is not None:
            return self.embeddings[self.pandas_codes[focus[0]] == focus[1]]
        else:
            return self.embeddings

    def retrieve_embeddings(self):
        embedding_file = osp.join(self.path, self.EMBEDDINGS_FILE_NAME.format(
            model_name=self.cfg["embedding_model"].model))
        
        if osp.exists(embedding_file):
            logger.info("Loading embeddings from file")
            self.embeddings = np.load(embedding_file)
        else:
            logger.info("Calculating embeddings with model: " +
                        str(self.cfg["embedding_model"]))
            
            self.embeddings = np.array(hydra.utils.instantiate(
                self.cfg["embedding_model"]).embed_documents(self.pandas_codes["content"]))
            np.save(embedding_file, self.embeddings)

        logger.info("Embeddings shape: " + str(self.embeddings.shape))