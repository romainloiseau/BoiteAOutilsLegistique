from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import re
import pandas as pd
import docx

import logging
logger = logging.getLogger(__name__)

MIN_TITLE_SIZE = 5
MIN_P_SIZE = 1
DONT_SKIP_LINE_P_SIZE = 5

INF = 9999999999999999

SUB_ALINEA = "sub-alinea"
SUB_ALINEA_ARTICLE = "sub-alinea-article"
SUB_ALINEA_ARTICLE_ALINEA = "sub-alinea-article-alinea"
ALINEA = "alinea"
ARTICLE = "article"
LEVELS = ["partie", "titre", "chapitre", "section", "sous-section"]
ANNEX_LEVELS = ["annexe"]
IDS = LEVELS + ANNEX_LEVELS + [ARTICLE, ALINEA]

NUMBERS_FORMAT = {
    "romain": [f"XXXX {i}" for i in ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv", "xxvi", "xxvii", "xxviii", "xxix", "xxx"]],
    "entiers": [f"XXXX {i}" for i in range(1, 100)],
    "adjectifs_f": [f"{i} XXXX" for i in ["première", "deuxième", "troisième", "quatrième", "cinquième", "sixième", "septième", "huitième", "neuvième", "dixième", "onzième", "douzième"]],
    "adjectifs_m": [f"{i} XXXX" for i in ["premier", "deuxième", "troisième", "quatrième", "cinquième", "sixième", "septième", "huitième", "neuvième", "dixième", "onzième", "douzième"]],
}

ADJ = ["bis", "ter", "quater", "quinquies", "sexies", "septies", "octies", "nonies", "decies", "undecies", "duodecies", "terdecies", "quaterdecies", "quindecies", "sexdecies", "septdecies", "octodecies", "novodecies", "vicies", "unvicies", "duovicies", "tervicies", "quatervicies", "quinvicies", "sexvicies", "septvicies", "octovicies", "novovicies", "tricies", "untricies", "duotricies", "tertricies", "quatertricies", "quintricies", "sextricies", "septtricies", "octotricies", "novotricies", "quadragies", "unquadragies", "duoquadragies", "terquadragies", "quaterquadragies", "quinquadragies", "sexquadragies", "septquadragies", "octoquadragies", "novoquadragies", "quinquagies", "unquinquagies", "duoquinquagies", "terquinquagies", "quaterquinquagies", "quinquinquagies", "sexquinquagies", "septquinquagies", "octoquinquagies", "novoquinquagies", "sexagies", "unsexagies", "duosexagies", "tersexagies", "quatersexagies", "quinsexagies", "sexsexagies", "septsexagies", "octosexagies", "novosexagies", "septuagies", "unseptuagies", "duoseptuagies", "terseptuagies", "quaterseptuagies", "quinseptuagies", "sexseptuagies", "septseptuagies", "octoseptuagies", "novoseptuagies", "octogies", "unoctogies", "duooctogies", "teroctogies", "quateroctogies", "quinoctogies", "sexoctogies", "septoctogies", "octooctogies", "novooctogies", "nonagies", "unnonagies", "duononagies", "ternonagies", "quaternonagies", "quinnonagies", "sexnonagies", "septnonagies", "octononagies", "novononagies", "centies"]

def a_smaller_than_b(a, b):
    if len(a.split(" ")) == 1:
        if a == "premier":
            a = "1"
        assert a.isnumeric()
        if len(b.split(" ")) == 1:
            if b == "premier":
                b = "1"
            assert b.isnumeric(), repr(b)
            return int(a) < int(b)
        else:
            assert len(b.split(" ")) == 2
            bnum, badd = b.split(" ")
            assert bnum.isnumeric()
            return int(a) <= int(bnum)
    else:
        assert len(a.split(" ")) == 2
        anum, aadd = a.split(" ")
        if anum == "premier":
            anum = "1"
        assert anum.isnumeric()
        if len(b.split(" ")) == 1:
            if b == "premier":
                b = "1"
            assert b.isnumeric()
            return int(anum) < int(b)
        else:
            assert len(b.split(" ")) == 2
            bnum, badd = b.split(" ")
            if bnum == "premier":
                bnum = "1"
            assert bnum.isnumeric()
            return int(anum) < int(bnum) or ((int(anum) == int(bnum)) and (ADJ.index(aadd) < ADJ.index(badd)))
        
def b_equals_one_plus_a(a, b):
    if len(a.split(" ")) == 1:
        if a == "premier":
            a = "1"
        assert a.isnumeric()
        if len(b.split(" ")) == 1:
            if b == "premier":
                b = "1"
            assert b.isnumeric(), repr(b)
            return int(a) + 1 == int(b)
        else:
            assert len(b.split(" ")) == 2
            bnum, badd = b.split(" ")
            assert bnum.isnumeric()
            return (int(a) == int(bnum)) and (badd == "bis")
    else:
        assert len(a.split(" ")) == 2
        anum, aadd = a.split(" ")
        if anum == "premier":
            anum = "1"
        assert anum.isnumeric()
        if len(b.split(" ")) == 1:
            if b == "premier":
                b = "1"
            assert b.isnumeric()
            return int(anum) + 1 == int(b)
        else:
            assert len(b.split(" ")) == 2
            bnum, badd = b.split(" ")
            if bnum == "premier":
                bnum = "1"
            assert bnum.isnumeric()
            return ((int(anum) == int(bnum)) and (ADJ.index(aadd) + 1 == ADJ.index(badd)))

def clean_paragraph(p):
    for c in ["\n", "\xa0", "\t", "◄", "  "]:
        p = p.replace(c, " ")
    p = p.strip()
    p = p.replace("novquagies", "novoquadragies")
    p = p.replace("quinquesquadragies", "quinquadragies")
    return p

def keep_paragraph(p):
    return p != "" and p != "\n" \
        and not p.startswith("►") \
        and not p.startswith("▼") \
        and not p.startswith("*") \
        and "".join(p.split('_')) != ""

def remove_useless(soup):
    title = None
    element = soup.find("title")
    if element:
        title = element.text
        element.decompose()
        
    element = soup.find("div", id="TOCSidebarWrapper")
    if element:
        element.decompose()

    element = soup.find("p", {"class": "disclaimer"})
    if element:
        element.decompose()

    element = soup.find("p", {"class": "reference"})
    if element:
        element.decompose()

    for element in soup.find_all("p", {"class": "arrow"}):
        element.decompose()

    return title

def read_html_lines(filename):
    with open(filename, "r") as f:
        soup = BeautifulSoup(f.read(), features="html.parser")
    
    title = remove_useless(soup)

    souptext = soup.text

    souptext = re.sub(' +', ' ', souptext)
    souptext = souptext.replace("\n\n", "\n")
    souptext = souptext.split("\n")

    text = []
    for p in souptext:
        ptext = clean_paragraph(p)
        if keep_paragraph(ptext):
            text.append(ptext)

    return text

def read_docx_lines(document):

    text = []
    for p in document.paragraphs:
        ptext = clean_paragraph(p.text)

        ptext = ptext.replace("▌", "")

        if keep_paragraph(ptext):
            text.append(ptext)

    return text

def get_levels_numbers(text):
    levels_numbers = {}

    line = 0
    while (len(levels_numbers) < len(LEVELS + ANNEX_LEVELS)) and (line < len(text)):

        if text[line].lower().startswith("titre"):
            print("text[line]", text[line])

        for k in LEVELS + ANNEX_LEVELS:
            if k not in levels_numbers.keys():
                for nf, kf in NUMBERS_FORMAT.items():
                    if text[line].lower().startswith(kf[0].replace("XXXX", k)):
                        levels_numbers[k] = [f.replace("XXXX", k) for f in kf]
                        line += 1
                        break

        line += 1

    print("levels_numbers", levels_numbers)

    return levels_numbers

def end_text_formula(paragraph):
    return paragraph.startswith("Fait à") and (", le" in paragraph)

def start_text_formula(paragraph):
    return "ONT ADOPTÉ LA PRÉSENTE DIRECTIVE:" in paragraph

def read_eu_text_from_lines(text, start_formula = None, end_formula = None):
    type, content = [], []

    started_formula = start_formula is None #False
    started = False

    counter_article = "0"
    counter_alinea = 0

    levels_numbers = get_levels_numbers(text)

    line = 0

    while line < len(text):
        if not started_formula:
            if start_formula is not None and eval(start_formula, {"paragraph": text[line]}): #start_text_formula(text[line]):
                started_formula = True

        else:
            is_level = False

            if end_formula is not None and eval(end_formula, {"paragraph": text[line]}): #end_text_formula(text[line]):
                line = len(text)
                break

            for current_level in levels_numbers.keys():

                level_text = " ".join(text[line].split(" ")[:3]) if ((len(text[line].split(" ")) > 2) and (text[line].split(" ")[2] in ADJ)) else " ".join(text[line].split(" ")[:2])
                level_text_get_next_line = text[line] == level_text

                if len(level_text.split(" ")) == 2:
                    is_current_level = level_text.lower() in levels_numbers[current_level]

                    if is_current_level and (text[line + 1] in ADJ):
                        text[line] = f"{text[line]} {text[line + 1]}"
                        text = text[:line+1] + text[line+2:]
                elif len(level_text.split(" ")) == 3:
                    is_current_level = (" ".join(level_text.lower().split(" ")[:2]) in levels_numbers[current_level]) and (level_text.lower().split(" ")[-1] in ADJ)
                else:
                    is_current_level = False

                is_current_level = is_current_level and (text[line + 1].lower() not in levels_numbers[current_level]) and (len(text[line + 1]) > MIN_TITLE_SIZE)

                if is_current_level:
                    is_level = True
                    started = True
                    
                    if current_level in ANNEX_LEVELS:
                        line = len(text)
                        break

                    type.append(current_level)

                    if level_text_get_next_line:
                        content.append(f"{text[line]} {text[line + 1]}")
                        line += 1
                    else:
                        content.append(f"{text[line]}")
                    
                    break

            if started and (not is_level):

                article_text = " ".join(text[line].split(" ")[:3]) if ((len(text[line].split(" ")) > 2) and (text[line].split(" ")[2] in ADJ)) else " ".join(text[line].split(" ")[:2])
                article_text_get_next_line = text[line] == article_text

                if (len(article_text.split(" ")) in [2, 3]) and article_text.lower().startswith(ARTICLE) and b_equals_one_plus_a(str(counter_article), str(" ".join(article_text.split(" ")[1:]))):
                    type.append(ARTICLE)
                    if article_text_get_next_line:
                        content.append(f"{text[line]} - {text[line + 1]}")
                        line += 1
                    else:
                        content.append(text[line])

                    counter_article = " ".join(article_text.split(" ")[1:])
                    counter_alinea = 0
                else:
                    if text[line].split(".")[0].isnumeric() and text[line].startswith(f'{text[line].split(".")[0]}.') and int(text[line].split(".")[0]) > counter_alinea:
                        type.append(ALINEA)
                        content.append(f"{text[line]}")
                        counter_alinea = int(text[line].split(".")[0])
                    elif type[-1] == ARTICLE:
                        type.append(ALINEA)
                        content.append(f"{text[line]}")
                        counter_alinea = INF
                    else:
                        while text[line].endswith("(") and text[line + 2][0].startswith(")") and text[line + 1].isnumeric():
                            text[line] = f"{text[line]}{text[line + 1]}{text[line + 2]}"
                            text = text[:line+1] + text[line+3:]

                        if len(text[line]) > MIN_P_SIZE:
                            if (content[-1] == f"{counter_alinea}.") or (len(content[-1].split("\n")[-1]) < DONT_SKIP_LINE_P_SIZE):
                                content[-1] = f"{content[-1]} {text[line]}"
                            else:
                                content[-1] = f"{content[-1]}\n{text[line]}"

        line += 1

    return split_long_paragraphs(type, content)
    
def split_long_paragraphs(type, content):

    new_type, new_content = [], []

    for t, c in zip(type, content):
        if t == "alinea" and c.split("\n")[0].endswith(f"comme suit:") and c.split("\n")[1].startswith("1) "):
            counter_sub_alinea = 0
            pre_sub_alinea = c.split("\n")[0]

            for sub_alinea in c.split("\n")[1:]:
                if sub_alinea.startswith(f"{counter_sub_alinea + 1}) "):
                    new_type.append(SUB_ALINEA)
                    new_content.append(f"{pre_sub_alinea}\n{sub_alinea}")
                    counter_sub_alinea += 1
                else:
                    new_content[-1] += f"\n{sub_alinea}"
        else:
            new_type.append(t)
            new_content.append(c)

    new_new_type, new_new_content = [], []
    for t, c in zip(new_type, new_content):

        if "\n" in c and (c.split("\n")[1].endswith("sont insérés:") or c.split("\n")[1].endswith("est inséré:")) and any([cc.startswith('"Article') or cc.startswith('Article') for cc in c.split("\n")]):

            pre_sub_alinea_article = "\n".join(c.split("\n")[:2])
            start = 2

            while not (c.split("\n")[start].startswith('"Article') or c.split("\n")[start].startswith("Article")):
                pre_sub_alinea_article += "\n" + c.split('\n')[start]
                start += 1
                advance = True
                print("start", c.split("\n")[start], start)
            
            for sub_alinea_article in c.split("\n")[start:]:
                if sub_alinea_article.startswith(f'"Article') or sub_alinea_article.startswith(f'Article'):
                    new_new_type.append(SUB_ALINEA_ARTICLE)
                    new_new_content.append(f"{pre_sub_alinea_article}\n{sub_alinea_article}")
                else:
                    new_new_content[-1] += f"\n{sub_alinea_article}"

        else:
            new_new_type.append(t)
            new_new_content.append(c)

    final_type, final_content = [], []

    for t, c in zip(new_new_type, new_new_content):
        if t == SUB_ALINEA_ARTICLE:

            start = 2
            while not (c.split("\n")[start].startswith("Article") or c.split("\n")[start].startswith('"Article')):
                start += 1


            if len(c.split("\n")[start].split(" ")) < 4:
                pre_sub_alinea_article_alinea = "\n".join(c.split("\n")[:start + 1]) + " - " + c.split("\n")[start + 1]
                start += 1
            else:
                pre_sub_alinea_article_alinea = "\n".join(c.split("\n")[:start + 1])

            counter_alinea = 0

            for sub_alinea_article_alinea in c.split("\n")[start:]:
                if sub_alinea_article_alinea.startswith(f"{counter_alinea + 1}. "):
                    final_type.append(SUB_ALINEA_ARTICLE_ALINEA)
                    final_content.append(f"{pre_sub_alinea_article_alinea}\n{sub_alinea_article_alinea}")
                    counter_alinea += 1
                else:
                    final_content[-1] += f"\n{sub_alinea_article_alinea}"
        else:
            final_type.append(t)
            final_content.append(c)

    return pd.DataFrame({"id": final_type, "text": final_content})

def read_html(filename, start_formula = None, end_formula = None):
    text = read_html_lines(filename)
    return read_eu_text_from_lines(text, start_formula = start_formula, end_formula = end_formula)

def read_docx(filename, start_formula = None, end_formula = None):
    return read_docx_from_document(docx.Document(filename), start_formula = start_formula, end_formula = end_formula)

def read_docx_from_document(document, start_formula = None, end_formula = None):
    text = read_docx_lines(document)
    return read_eu_text_from_lines(text, start_formula = start_formula, end_formula = end_formula)

def read_text(filename, start_formula = None, end_formula = None):
    if filename.endswith(".html"):
        return read_html(filename, start_formula = start_formula, end_formula = end_formula)
    elif filename.endswith(".docx"):
        return read_docx(filename, start_formula = start_formula, end_formula = end_formula)
    else:
        raise ValueError