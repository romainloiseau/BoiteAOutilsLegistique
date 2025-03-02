# 📖 Boite à outils légistique

Cette application a été développée pour simplifier l'analyse et la gestion des textes juridiques français et européens en utilisant des modèles d'IA avancés.

## ⚙️ Fonctionnalités principales

- 📜 **Gestion des textes de loi** : Importez des textes législatifs français et européens depuis des sources telles que Legifrance et EUR-Lex, et visualisez-les de manière claire et organisée.
- 🤖 **Analyse assistée par IA** : Utilisez des modèles de langage comme GPT-3.5-turbo et GPT-4 pour analyser et résumer des articles de loi complexes.
- 🛠️ **Transposition de textes** : Vérifiez la compatibilité des articles européens avec le droit français et identifiez les modifications législatives nécessaires.
- 📥 **Téléchargement de données** : Exportez vos analyses sous divers formats (```.csv```, ```.json```, ```.twt```, ```.html```) pour une utilisation ultérieure.

## 🛠️ Configuration

Le projet utilise les modèles de Langchain pour l'analyse textuelle et les embeddings. La configuration des modèles peut être ajustée dans le fichier de configuration comme suit :

- **Modèles de langage** : GPT-3.5-turbo, GPT-4, Llama3, Gemma2, Mistral, Mixtral.
- **Modèles d'embedding** : OpenAI Text Embeddings, BGE-M3.
- **Outils de clustering et de réduction de dimensions** : KMeans, t-SNE.

## 📥 Installation

Clonez ce dépôt ...

```bash
git clone https://github.com/romainloiseau/BoiteAOutilsLegistique.git
cd BoiteAOutilsLegistique
```

... et installez les dépendances :

```bash
conda env create -f environment.yml
conda activate legaltoolbox
```

_**Note:** We use `conda`, you can download it [here](https://www.anaconda.com/download)._

🔑 Ajoutez ensuite vos clés API dans le fichier de configuration ```configs/perso/defaults.yaml```.

### LLMs

#### Local

Cette implémentation utilise [`LangChain`](https://python.langchain.com/docs/get_started/quickstart) ainsi que [`Ollama`](https://ollama.com/) pour exécuter localement des modèles de langage open-source, et l'[API d'`OpenAI`](https://openai.com/api/) pour exécuter des modèles en ligne.

Avant d'utiliser cette implémentation via ollama, vous devrez télécharger les poids des modèles que vous utiliserez. Pour ce faire, vous pouvez utiliser la commande `ollama pull modelname`.

## 🚀 Usage

Pour exécuter ce implémentation localement, vous devrez lancer un serveur `Ollama` local en utilisant la commande `ollama serve` dans un terminal. Ce serveur communiquera avec votre code et votre GPU. Si vous utilisez l'API d'`OpenAI`, cela n'est pas nécessaire.

Remplissez également le fichier `configs/perso/defaults.yaml` afin de préciser les clés `legifrance`, `openai` et `langsmith` au besoin.

Ensuite, lancez l'application avec :

```bash
streamlit run Accueil.py
```

Pour utiliser l'application, ouvrez le lien sur lequel l'application est lancée (en principe http://localhost:8501), puis :

1. 📂 **Sélectionnez un texte** : Importez un texte législatif français ou européen ou sélectionnez-en un existant.
2. 🔧 **Choisissez l'outil d'analyse** : Selon le type de texte, plusieurs outils s'offrent à vous (résumé, projection, analyse d'impact, tableau de transposition).
3. ⚡ **Lancez l'analyse** : L'application analysera automatiquement le texte sélectionné et proposera des résumés ou des suggestions de transposition.
4. 📊 **Téléchargez ou visualisez les résultats** : Visualisez les résultats dans l'application ou exportez-les en ```.csv```, ```.json```, ```.twt``` ou ```.html```.

## ⚠️ Avertissements

- Certaines analyses peuvent nécessiter un temps de calcul important et entraîner des coûts supplémentaires en fonction de l'API utilisée (OpenAI, Llama, etc.). Faites attention à l'utilisation des LLMs pour éviter des coûts inattendus.