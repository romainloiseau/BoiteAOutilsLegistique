from .base import BaseData
import logging
from hydra.utils import to_absolute_path
import os.path as osp
import numpy as np
import pandas as pd
import hydra
import os
import streamlit as st
import shutil
from legaltoolbox.utils import read_eu

from hydra.utils import instantiate

logger = logging.getLogger(__name__)

NO_TRANSPOSITION = {}

class EUData(BaseData):
    INPUT_FILE_NAME = ["input.html", "input.docx"]
    CONTENT_FILE_NAME_wo_ext = "content"

    CONTENT_ID = read_eu.ALINEA
    ARTICLE_TITLE_ID = read_eu.ARTICLE
    IDS = read_eu.IDS
    HEAD_IDS = read_eu.LEVELS + [ARTICLE_TITLE_ID]

    def __init__(self, text = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.path = to_absolute_path(
            osp.join(self.cfg["path"], self.cfg["folder"]))
        
        if not osp.exists(self.path):
            os.makedirs(self.path)

        self.name = self.cfg["folder"]

        if text is not None:
            if type(text) is st.runtime.uploaded_file_manager.UploadedFile:
                with open(osp.join(self.path, "input.docx"), "wb") as f:
                    f.write(text.getvalue())
            else:
                with open(osp.join(self.path, "input.html"), "w") as f:
                    f.write(text)

        self.success = False
        self.retrieve_content()
        self.retrieve_embeddings()

    def show_in_streamlit(self, show_content, key):
        st.markdown(f"## {self.name}")

        if instantiate(show_content, key=key):
            st.write(self.content)

    def download_csv(self):
        return osp.join(self.path, self.CONTENT_FILE_NAME_wo_ext + ".csv")
    
    def download_txt(self):
        return osp.join(self.path, "articles.txt")
    
    def download_html(self):
        return osp.join(self.path, "articles.html")

    def add_peripheral_text(self, peripheral_text, peripheral_text_name, keep_order=False):
        raise NotImplementedError
        peripheral_text_embedding_path = osp.join(self.path, "peripheral_" + peripheral_text_name + "_" + self.EMBEDDINGS_FILE_NAME.format(model_name=self.cfg["embedding_model"].model))
        peripheral_text_path = osp.join(self.path, "peripheral_" + peripheral_text_name + ".txt")

        if osp.exists(peripheral_text_path):
            logger.info("Peripheral text already added")

        if type(peripheral_text) == dict:
            text = [v for k, v in peripheral_text.items()]
        elif type(peripheral_text) == list:
            text = peripheral_text
        elif type(peripheral_text) == str:
            text = [peripheral_text]
        else:
            raise ValueError(f"Type {type(peripheral_text)} not supported")

        self.retrieve_articles_embeddings()

        if osp.exists(peripheral_text_embedding_path):
            peripheral_text_embeddings = np.load(peripheral_text_embedding_path)
        else:
            peripheral_text_embeddings = np.array(hydra.utils.instantiate(
                self.cfg["embedding_model"]).embed_documents(text))
            np.save(peripheral_text_embedding_path, peripheral_text_embeddings)

            with open(peripheral_text_path, "w") as f:
                f.write("\n\n\n".join(text))

        distance = np.expand_dims(self.articles_embeddings, 0) - np.expand_dims(peripheral_text_embeddings, 1)
        distance = (distance ** 2).sum(axis=-1) ** 0.5

        import matplotlib.pyplot as plt
        plt.imshow(distance)
        plt.savefig(osp.join(self.path, "distance.png"))

    def retrieve_content(self):
        if hasattr(self, "content"):
            return
        
        elif osp.exists(osp.join(self.path, self.CONTENT_FILE_NAME_wo_ext + ".csv")):
            self.content = pd.read_csv(osp.join(self.path, self.CONTENT_FILE_NAME_wo_ext + ".csv"))

        else:
            input_file = None
            for input_file_name in self.INPUT_FILE_NAME:
                if osp.exists(osp.join(self.path, input_file_name)):
                    input_file = osp.join(self.path, input_file_name)

            assert input_file is not None, f"Input file does not exist"

            self.content = read_eu.read_text(input_file, start_formula=self.cfg["start_formula"], end_formula=self.cfg["end_formula"])
            

            self.content_to_csv()

        self.success = len(self.content) > 0
        self.analyse_content()

    def delete_folder_and_return_none(self):
        shutil.rmtree(self.path)
        return None

    def analyse_content(self):
        if osp.exists(osp.join(self.path, "articles.txt")):
            with open(osp.join(self.path, "articles.txt"), "r") as f:
                self.articles = f.read().split("\n\n\n")
        else:
            unique = np.unique(self.content["id"], return_counts=True)

            logger.info(
                "Unique ids: " + " / ".join([f"{u} ({c})" for u, c in zip(unique[0], unique[1])]))

            self.articles = []
            for c, text in zip(self.content["id"], self.content["text"]):
                if c == self.ARTICLE_TITLE_ID:
                    self.articles.append(text)
                elif c == self.CONTENT_ID:
                    self.articles[-1] += "\n" + text

            self.articles = np.array(self.articles)

            with open(osp.join(self.path, "articles.txt"), "w") as f:
                f.write("\n\n\n".join(self.articles))


            levels_open = {
                "partie": False,
                "titre": False,
                "chapitre": False,
                "section": False,
                "sous-section": False,
            }
            levels = read_eu.LEVELS


            self.articles_html = "<html><head><title>Summary</title></head><body>\n"
            i = 0
            while i < len(self.content):

                if self.content["id"][i] == "article":
                    self.articles_html += f"<li><details><summary>{self.content['text'][i]}</summary>\n<ul>\n"

                    i += 1

                    while (i < len(self.content)) and (self.content["id"][i] == "alinea"):
                        text_to_html = self.content['text'][i].replace('\n', '<br>\n')
                        self.articles_html += f"<li>{text_to_html}</li>\n"
                        i += 1
                    
                    self.articles_html += "</ul></details></li>\n"
                elif self.content["id"][i] in levels:
                    if levels_open[self.content["id"][i]]:
                        close_level = len(levels) - 1
                        while levels[close_level] != self.content["id"][i]:
                            if levels_open[levels[close_level]]:
                                self.articles_html += "</ul></details></li>\n"
                                levels_open[levels[close_level]] = False
                            close_level -= 1

                        self.articles_html += "</ul></details>\n"
                        if self.content["id"][i] != levels[0]:
                            self.articles_html += "</li>\n"

                        levels_open[self.content["id"][i]] = False
                    
                    levels_open[self.content["id"][i]] = True

                    if self.content["id"][i] != levels[0]:
                        self.articles_html += "<li>\n"
                    self.articles_html += f"<details><summary>{self.content['text'][i]}</summary>\n<ul>\n"

                    i += 1
            self.articles_html += "</body>"

            with open(osp.join(self.path, "articles.html"), "w") as f:
                f.write(self.articles_html)

    def retrieve_embeddings(self):

        if not hasattr(self, "embeddings"):
            embedding_file = osp.join(self.path, self.EMBEDDINGS_FILE_NAME.format(
                model_name=self.cfg["embedding_model"].model))
            if osp.exists(embedding_file):
                logger.info("Loading embeddings from file")
                self.embeddings = np.load(embedding_file)
            else:
                logger.info("Calculating embeddings with model: " +
                            str(self.cfg["embedding_model"]))
                self.embeddings = np.array(hydra.utils.instantiate(
                    self.cfg["embedding_model"]).embed_documents(self.get_texts(ids=self.IDS)))
                np.save(embedding_file, self.embeddings)

        assert len(self.embeddings) == len(
            self.content), f"Embeddings shape {self.embeddings.shape} does not match content shape {len(self.content)}"

        logger.info("Embeddings shape: " + str(self.embeddings.shape))

    def retrieve_articles_embeddings(self):

        if not hasattr(self, "articles_embeddings"):
            embedding_file = osp.join(self.path, "articles_" + self.EMBEDDINGS_FILE_NAME.format(
                model_name=self.cfg["embedding_model"].model))
            if osp.exists(embedding_file):
                logger.info("Loading embeddings from file")
                self.articles_embeddings = np.load(embedding_file)
            else:
                logger.info("Calculating embeddings with model: " +
                            str(self.cfg["embedding_model"]))
                self.articles_embeddings = np.array(hydra.utils.instantiate(
                    self.cfg["embedding_model"]).embed_documents(self.articles))
                np.save(embedding_file, self.articles_embeddings)

        assert len(self.articles_embeddings) == len(
            self.articles), f"Embeddings shape {self.articles_embeddings.shape} does not match articles shape {len(self.articles)}"

        logger.info("Embeddings shape: " + str(self.articles_embeddings.shape))

    def content_to_csv(self):
        self.content.to_csv(
            osp.join(self.path, self.CONTENT_FILE_NAME_wo_ext + ".csv"), index=False)

    def get_texts(self, ids=None, need_transposition=None):
        if ids is None:
            ids = self.CONTENT_ID

        if type(ids) is str:
            ids = [ids]

        for id in ids:
            assert id in self.IDS, f"Id {id} not in {self.IDS}"

        condition = self.content["id"].isin(ids)

        if need_transposition is not None:
            if need_transposition:
                condition = condition & (
                    self.content["need_transposition"] == 0)
            else:
                condition = condition & (
                    self.content["need_transposition"] != 0)

        return self.content[condition]["text"]

    def get_embeddings(self, ids=None, need_transposition=None):
        if ids is None:
            ids = self.CONTENT_ID

        if type(ids) is str:
            ids = [ids]

        for id in ids:
            assert id in self.IDS, repr(id)

        condition = self.content["id"].isin(ids)

        if need_transposition is not None:
            if need_transposition:
                condition = condition & (
                    self.content["need_transposition"] == 0)
            else:
                condition = condition & (
                    self.content["need_transposition"] != 0)

        return self.embeddings[condition]

    def is_transposition_needed(self, llm=None):
        need_transposition = []
        need_transposition_comment = []

        for c, text in zip(self.content["id"], self.content["text"]):
            if c != self.CONTENT_ID:
                need_transposition.append(1)
                need_transposition_comment.append("Not a content id")
            else:
                need_transposition.append(0)
                need_transposition_comment.append(None)

        self.content["need_transposition"] = need_transposition
        self.content["need_transposition_comment"] = need_transposition_comment

        logger.info(
            f"Keep content ids:\tRemoved {100*(np.mean(self.content['need_transposition'] != 0)):.2f}% rows, {np.sum(self.content['need_transposition'] == 0)} alineas left.")

        for row in self.content.itertuples():
            if row.need_transposition == 0:
                for key, value in NO_TRANSPOSITION.items():
                    for v in value:
                        if (v.lower() in row.text.lower()) or (v.lower().replace("'", "’") in row.text.lower()):
                            self.content.at[row.Index,
                                            "need_transposition"] = 2
                            self.content.at[row.Index,
                                            "need_transposition_comment"] = f"{key}: Found '{v}' in text"
                            break

        logger.info(
            f"Remove auto text:\tRemoved {100*(np.mean(self.content['need_transposition'] != 0)):.2f}% rows, {np.sum(self.content['need_transposition'] == 0)} alineas left.")

        # if llm is not None:
        #    llm = hydra.utils.instantiate(llm)
        #    for row in self.content.itertuples():
        #        if row.need_transposition:

        self.content_to_csv()

    def analyse_text(self):
        html = "<html><head><title>Projection</title></head><body>"

        # texts = self.get_texts()
        # embeddings = self.get_embeddings()
        # html = self.run_clustering_and_projection(html, embeddings, texts, texts)

        texts = self.get_texts(ids=self.ARTICLE_TITLE_ID)
        embeddings = self.get_embeddings(ids=self.ARTICLE_TITLE_ID)

        html = self.run_clustering_and_projection(
            html, embeddings, texts, self.articles)[0]

        html += "</body></html>"
        with open(osp.join(self.path, "projection.html"), "w") as f:
            f.write(html)

    def run_clustering(self, embeddings):
        distances = ((embeddings[1:] - embeddings[:-1])
                     ** 2).sum(axis=1) ** 0.5

        threshold = distances.max()

        texts_per_clusters = self.cfg["kmeans_texts_per_clusters"] if "kmeans_texts_per_clusters" in self.cfg else self.cfg["kmeans"]["texts_per_clusters"]
        k = int(len(embeddings) / texts_per_clusters)

        n_clusters_list = []
        for thresh in np.linspace(0, 10*threshold, 10000):

            labels, n_clusters = self.compute_clustering_for_threshold(
                embeddings, thresh, texts_per_clusters)

            n_clusters_list.append(n_clusters)

            if n_clusters <= k:
                cluster_centers = np.array(
                    [embeddings[labels == i].mean(axis=0) for i in range(n_clusters)])
                # import matplotlib.pyplot as plt
                # plt.plot(np.arange(len(n_clusters_list)), n_clusters_list)
                # plt.savefig("test.png")
                break

        return labels, cluster_centers, n_clusters

    def compute_clustering_for_threshold(self, embeddings, thresh, texts_per_clusters):
        labels = [0]  # np.zeros(len(embeddings), dtype=int)
        # labels[1:] = (distances > thresh).cumsum()

        current_embedding, n_current_embedding = embeddings[0], 1
        # , embedding in enumerate(embeddings[1:], 1):
        for i in range(1, len(embeddings)):
            # * n_current_embedding**.25:# n_current_embedding**-.5 or n_current_embedding >= 2*texts_per_clusters:
            if ((embeddings[i] - embeddings[i-1]) ** 2).sum() ** 0.5 < thresh / n_current_embedding**.1:
                labels.append(labels[-1])
                n_current_embedding += 1
            else:
                labels.append(labels[-1] + 1)
                # current_embedding = embedding
                n_current_embedding = 1

                # current_embedding = embedding
                # current_embedding = (current_embedding * (n_current_embedding - 1) + embedding) / n_current_embedding
        labels = np.array(labels)
        n_clusters = labels.max() + 1
        return labels, n_clusters
