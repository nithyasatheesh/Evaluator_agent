import streamlit as st
import pandas as pd
import zipfile
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from bs4 import BeautifulSoup
from docx import Document
from PyPDF2 import PdfReader
import nbformat
from pathlib import Path

st.set_page_config(
    page_title="Case Study Evaluator",
    layout="wide"
)

st.title("AI Case Study Evaluator")

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

####################################
# FILE READERS
####################################

def read_pdf(file):

    try:
        reader = PdfReader(file)

        text = []

        for page in reader.pages:
            t = page.extract_text()

            if t:
                text.append(t)

        return "\n".join(text)

    except:
        return ""

def read_docx(file):

    try:

        doc = Document(file)

        return "\n".join(
            p.text
            for p in doc.paragraphs
        )

    except:
        return ""

def read_html(raw):

    soup = BeautifulSoup(
        raw,
        "html.parser"
    )

    for tag in soup(
        ["script","style"]
    ):
        tag.decompose()

    return soup.get_text()

def read_ipynb(raw):

    try:

        nb = nbformat.reads(
            raw,
            as_version=4
        )

        output=[]

        for cell in nb.cells:

            if cell.cell_type=="markdown":

                output.append(
                    "".join(cell.source)
                )

            elif cell.cell_type=="code":

                output.append(
                    "CODE:\n"+
                    "".join(cell.source)
                )

        return "\n".join(output)

    except:
        return ""

def summarize_csv(raw):

    try:

        df=pd.read_csv(
            io.StringIO(raw)
        )

        return f"""
Rows:{df.shape[0]}
Columns:{list(df.columns)}
Head:
{df.head(3)}
"""

    except:

        return ""

####################################
# RUBRIC
####################################

def rubric_text(df):

    return "\n".join(

        f"""
Criterion:
{r['Criterion']}

Max Score:
{r['Max Score']}

Description:
{r['Description']}
"""

        for _,r
        in df.iterrows()
    )

####################################
# ZIP PARSER
####################################

def parse_submission(
    zip_bytes
):

    submission={

        "docs":[],

        "code":[],

        "notebooks":[],

        "datasets":[],

        "sql":[],

        "frontend":[],

        "backend":[],

        "images":[]

    }

    z=zipfile.ZipFile(
        io.BytesIO(zip_bytes)
    )

    for path in z.namelist():

        try:

            raw=z.read(path)

            decoded=raw.decode(
                errors="ignore"
            )

            suffix=Path(
                path
            ).suffix.lower()

            if suffix==".pdf":

                submission["docs"].append(
                    read_pdf(
                        io.BytesIO(raw)
                    )
                )

            elif suffix==".docx":

                submission["docs"].append(
                    read_docx(
                        io.BytesIO(raw)
                    )
                )

            elif suffix in [
                ".html",
                ".htm"
            ]:

                submission["docs"].append(
                    read_html(decoded)
                )

            elif suffix==".py":

                if "frontend" in path:

                    submission[
                        "frontend"
                    ].append(
                        decoded
                    )

                elif "backend" in path:

                    submission[
                        "backend"
                    ].append(
                        decoded
                    )

                else:

                    submission[
                        "code"
                    ].append(
                        decoded
                    )

            elif suffix==".ipynb":

                submission[
                    "notebooks"
                ].append(
                    read_ipynb(
                        decoded
                    )
                )

            elif suffix==".csv":

                submission[
                    "datasets"
                ].append(
                    summarize_csv(
                        decoded
                    )
                )

            elif suffix==".sql":

                submission[
                    "sql"
                ].append(
                    decoded
                )

            elif suffix==".md":

                submission[
                    "docs"
                ].append(
                    decoded
                )

            elif suffix in [
                ".png",
                ".jpg",
                ".jpeg"
            ]:

                submission[
                    "images"
                ].append(path)

        except:
            pass

    return submission

####################################
# CONTEXT
####################################

def build_context(
    submission
):

    return f"""

DOCUMENTATION

{' '.join(submission['docs'])[:12000]}

NOTEBOOKS

{' '.join(submission['notebooks'])[:10000]}

BACKEND

{' '.join(submission['backend'])[:10000]}

FRONTEND

{' '.join(submission['frontend'])[:7000]}

CODE

{' '.join(submission['code'])[:7000]}

DATABASE

{' '.join(submission['sql'])[:5000]}

DATASETS

{submission['datasets']}

IMAGES

{submission['images']}

"""

####################################
# OPENAI
####################################

def evaluate(
    prompt
):

    response=client.chat.completions.create(

        model="gpt-4.1",

        temperature=0,

        messages=[

            {

                "role":"system",

                "content":"""
You are a strict evaluator.

Evaluate complete submission.

Never score files separately.

Return JSON only.
"""
            },

            {

                "role":"user",

                "content":prompt

            }

        ]

    )

    return response.choices[
        0
    ].message.content

####################################
# UI
####################################

problem=st.file_uploader(
"Problem",
["pdf","docx"]
)

rubric=st.file_uploader(
"Rubric",
["xlsx"]
)

zip_files=st.file_uploader(

"Participant ZIP",

type=["zip"],

accept_multiple_files=True

)

custom=st.text_area(
"Extra Prompt"
)

if st.button(
"Evaluate"
):

    rubric_df=pd.read_excel(
        rubric
    )

    rubric_str=rubric_text(
        rubric_df
    )

    if problem.name.endswith(
        ".pdf"
    ):

        problem_text=read_pdf(
            problem
        )

    else:

        problem_text=read_docx(
            problem
        )

    results=[]

    def process(z):

        submission=parse_submission(
            z.read()
        )

        context=build_context(
            submission
        )

        prompt=f"""

PROBLEM

{problem_text}

RUBRIC

{rubric_str}

CUSTOM

{custom}

SUBMISSION

{context}

Return:

{{
"scores":{{}},
"strengths":[],
"improvements":[]
}}

"""

        raw=evaluate(
            prompt
        )

        return {

            "participant":
            z.name,

            "evaluation":
            raw

        }

    with ThreadPoolExecutor(
        max_workers=4
    ) as ex:

        results=list(
            ex.map(
                process,
                zip_files
            )
        )

    out=pd.DataFrame(
        results
    )

    st.dataframe(out)

    excel=io.BytesIO()

    out.to_excel(
        excel,
        index=False
    )

    st.download_button(

        "Download",

        excel,

        "results.xlsx"

    )
