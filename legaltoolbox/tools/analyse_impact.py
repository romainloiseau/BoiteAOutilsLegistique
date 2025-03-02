import streamlit as st
import plotly.graph_objects as go
import numpy as np
from tqdm.auto import tqdm
from langchain_community.vectorstores import FAISS

import pandas as pd

from pylegifrance.models.consult import GetArticle
from pylegifrance import LegiHandler

from legaltoolbox.utils import plotly as myplotly

@st.cache_data(show_spinner = False)
def run_analyse_impact(_cfg, fr_embeddings: np.array, _fr_texts: list, eu_embeddings: np.array, html_save_path: str) -> go.Figure:

    k = _cfg.tools.analyse_impact.k_relevant_articles_per_alinea
    fetch_k = _cfg.tools.analyse_impact.fetch_k_relevant_articles_per_alinea
    len_eu = len(eu_embeddings)
    top_percent = _cfg.tools.analyse_impact.top_percent

    uids_counts = pd.DataFrame(columns=["page_content", "uid", "id", 'to_ids'] + [f"count_{i}" for i in range(k)])
    
    uids_counts["uid"] = [text.metadata["uid"] for text in _fr_texts]
    uids_counts["to_ids"] = [text.metadata["to_ids"] for text in _fr_texts]
    uids_counts["id"] = [text.metadata["id"] for text in _fr_texts]
    uids_counts["page_content"] = [text.page_content for text in _fr_texts]
    uids_counts = uids_counts.fillna(0)
    
    text_embedding_pairs = zip([text.page_content for text in _fr_texts], fr_embeddings)
    retriever = FAISS.from_embeddings(
        text_embeddings = text_embedding_pairs,
        embedding = _cfg.embedding_model,
        metadatas=[text.metadata for text in _fr_texts]
    )

    progress_bar = st.progress(0)
    scores = []
    retrieved = []
    for iemb, emb in tqdm(enumerate(eu_embeddings)):
        retrieved.append(retriever.similarity_search_with_score_by_vector(emb, k=k, fetch_k=fetch_k))

        scores.append(np.array([o[1] for o in retrieved[-1]]))

        progress_bar.progress((iemb + 1) / len_eu)
    progress_bar.empty()

    scores = np.array(scores)
    threshold = np.mean(scores) + .5 * np.std(scores)
    
    out_ai_assistant = []
    out_ai_context = []
    n_ai_assistant = []

    progress_bar = st.progress(0)
    for iemb, r in tqdm(enumerate(retrieved)):
        retrieved_kept = [o[0] for o in r if o[1] < threshold]

        n_ai_assistant.append(len(retrieved_kept))
        
        out_ai_assistant.append([(o.metadata["uid"], o.metadata["id"]) for o in retrieved_kept])
        out_ai_context.append("\n\n".join([f"{o.metadata['uid']} : {o.page_content}" for o in retrieved_kept]))

        for i, o in enumerate(retrieved_kept):
            uids_counts.loc[uids_counts["uid"] == o.metadata["uid"], f"count_{i}"] += 1
                
        progress_bar.progress((iemb + 1) / len_eu)
    progress_bar.empty()

    fig = go.Figure()
    sums = uids_counts[[f"count_{i}" for i in range(k)]].sum(axis=1)
    kernel = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
    kernel /= kernel.sum()

    smoothed = np.convolve(sums, kernel, mode='same')
    smoothed *= smoothed > 2

    hovertext_h = np.array([" ".join(uid.split(" ")[1:]) for uid in uids_counts["uid"]])
    hovertext = np.array([f"<b>{' '.join(row['uid'].split(' ')[1:])}</b><br>{row['page_content']}" for _, row in uids_counts.iterrows()])
    hovertext_page_content = np.array(uids_counts["page_content"].to_list())
    x_to_ids = np.array(uids_counts["to_ids"].to_list())
    ids = np.array(uids_counts["id"].to_list())
    keep = smoothed != 0
    
    hovertext, hovertext_h, hovertext_page_content, smoothed, sums, x_to_ids, ids = hovertext[keep], hovertext_h[keep], hovertext_page_content[keep], smoothed[keep], sums[keep], x_to_ids[keep], ids[keep]
            
    fig.add_trace(go.Scatter(
        x=hovertext_h, y=smoothed, mode='lines', line=dict(color='red'), 
        name='Count', fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)',
        hoverinfo="text", hovertext=myplotly.format_text_for_plotly(hovertext)
        ))

    sorted_smoothed = np.argsort(sums)[::-1]
    top10 = np.sort(sorted_smoothed[:int(len(sums) * top_percent)])

    main_texts = "\n\n".join([f"{uid} : {uidtext}" for uid, uidtext in zip(hovertext[top10], hovertext_page_content[top10])])
    
    fig.add_trace(go.Scatter(
        x=hovertext_h[top10], y=smoothed[top10], marker=dict(color='blue'),
        name=f'Top {100*top_percent:0.0f}%', mode='markers',
        textposition='top center', hoverinfo="text", hovertext=myplotly.format_text_for_plotly(hovertext[top10])
        ))

    x_to_ids = np.array(sum([eval(x_to_id) for x_to_id in x_to_ids], []))
    x_to_ids_unique, x_to_ids_count = np.unique(x_to_ids, return_counts=True)
    keep = x_to_ids_count > np.mean(x_to_ids_count) + np.std(x_to_ids_count)
    x_to_ids_unique, x_to_ids_count = x_to_ids_unique[keep], x_to_ids_count[keep]

    fig.add_trace(get_rank_2(_cfg, ids, top10, x_to_ids_unique, x_to_ids_count, uids_counts, main_texts))
    
    fig.update_layout(barmode='group', xaxis_tickangle=-45)

    html = f"<h2>Top {100*top_percent:0.0f}% des articles les plus impactés</h2>"
    html += "<br><br>".join(hovertext[top10])

    with open(html_save_path, "w") as f:
        f.write(f"""
            <h1>Analyse de l'impact du texte européen sur le droit français</h1>
            \n<br>\n
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
            \n<br><br>\n
            {html}
        """)

    return fig, html, {"main_texts": main_texts, "uids_counts": uids_counts, "out_ai_assistant": out_ai_assistant, "out_ai_context": out_ai_context}

def get_rank_2(_cfg, ids, top10, x_to_ids_unique, x_to_ids_count, uids_counts, main_texts):
    client = LegiHandler()
    client.set_api_keys(
        legifrance_api_key = _cfg.perso.legifrance_api_key,
        legifrance_api_secret = _cfg.perso.legifrance_api_secret,
    )

    hover_text_rank2, smoothed_rank2 = [], []
    for to_id, to_id_count in zip(x_to_ids_unique, x_to_ids_count):
        if to_id not in ids[top10]:
            if to_id in uids_counts["id"].values:
                article = uids_counts[uids_counts["id"] == to_id]
                main_texts += f"\n\n{article['uid'].values[0]} : {article['page_content'].values[0]}"
                hover_text_rank2.append(article["uid"].values[0])
                smoothed_rank2.append(to_id_count)
            else:
                art = GetArticle(id=to_id)
                article = client.call_api(route=art.Config.route, data=art.model_dump(mode="json")).json()["article"]
                print("NEW TEXT", to_id)
                if article is not None:
                    if article["etat"] == "VIGUEUR":
                        print("EN VIGUEUR")
                        print(article["num"])
                        print(article["context"]["titreTxt"][0]["titre"])
                        print(article["texte"])
                        article_du_code = f"Article {article['num']} du {article['context']['titreTxt'][0]['titre']}"
                        main_texts += f"\n\n{article_du_code} : {article['texte']}"
                        hover_text_rank2.append(article_du_code)
                        smoothed_rank2.append(to_id_count)
                    elif article["etat"] in ["TRANSFERE", "MODIFIE"]:
                        for articleversion in article['articleVersions']:
                            if articleversion["etat"] == "VIGUEUR":
                                art2 = GetArticle(id=articleversion["id"])
                                article2 = client.call_api(route=art2.Config.route, data=art2.model_dump(mode="json")).json()["article"]
                                print(f"{article['etat']} => EN VIGUEUR")
                                print(article2["num"])
                                print(article2["context"]["titreTxt"][0]["titre"])
                                print(article2["texte"])
                                article_du_code = f"Article {article2['num']} du {article2['context']['titreTxt'][0]['titre']}"
                                main_texts += f"\n\n{article_du_code} : {article2['texte']}"
                                hover_text_rank2.append(article_du_code)
                                smoothed_rank2.append(to_id_count)
                else:
                    print("Not found")

    return go.Scatter(x=hover_text_rank2, y=smoothed_rank2, marker=dict(color='green'), name='Rank 2', mode='markers', textposition='top center',  hoverinfo="text", hovertext=hover_text_rank2)