import streamlit as st
import pandas as pd
import zipfile
import io
import json
import re

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from bs4 import BeautifulSoup
from docx import Document
from PyPDF2 import PdfReader
import nbformat


# --------------------------------
# CONFIG
# --------------------------------

st.set_page_config(
    page_title="AI Case Study Evaluator",
    layout="wide"
)

st.title("📊 AI Case Study Evaluator")

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)


# --------------------------------
# READERS
# --------------------------------

def read_pdf(file):

    try:

        reader=PdfReader(file)

        pages=[]

        for p in reader.pages:

            txt=p.extract_text()

            if txt:

                pages.append(txt)

        return "\n".join(pages)

    except:

        return ""


def read_docx(file):

    try:

        doc=Document(file)

        return "\n".join(

            p.text

            for p in doc.paragraphs

        )

    except:

        return ""


def read_html(text):

    try:

        soup=BeautifulSoup(

            text,

            "html.parser"

        )

        for tag in soup(

            ["script","style"]

        ):

            tag.decompose()

        return soup.get_text(
            separator="\n"
        )

    except:

        return ""


def read_notebook(text):

    try:

        nb=nbformat.reads(

            text,

            as_version=4

        )

        output=[]

        for cell in nb.cells:

            if cell.cell_type=="markdown":

                output.append(

                    "".join(
                        cell.source
                    )

                )

            elif cell.cell_type=="code":

                output.append(

                    "CODE:\n"+

                    "".join(
                        cell.source
                    )

                )

        return "\n".join(output)

    except:

        return ""


def summarize_csv(text):

    try:

        df=pd.read_csv(

            io.StringIO(text)

        )

        return f"""

Rows:
{df.shape[0]}

Columns:
{list(df.columns)}

Sample:

{df.head(3).to_string()}

"""

    except:

        return ""


# --------------------------------
# RUBRIC
# --------------------------------

def rubric_to_text(df):

    output=[]

    for _,row in df.iterrows():

        output.append(

f"""
Criterion:
{row["Criterion"]}

Max Score:
{row["Max Score"]}

Description:
{row["Description"]}
"""

        )

    return "\n".join(output)


# --------------------------------
# ZIP PARSER
# --------------------------------

def parse_submission(zip_bytes):

    result={

        "documentation":[],

        "code":[],

        "notebooks":[],

        "database":[],

        "datasets":[],

        "images":[]

    }

    z=zipfile.ZipFile(

        io.BytesIO(zip_bytes)

    )

    for file in z.namelist():

        try:

            raw=z.read(file)

            suffix=Path(
                file
            ).suffix.lower()

            decoded=raw.decode(

                errors="ignore"

            )

            if suffix==".pdf":

                result[
                    "documentation"
                ].append(

                    read_pdf(
                        io.BytesIO(raw)
                    )

                )

            elif suffix==".docx":

                result[
                    "documentation"
                ].append(

                    read_docx(
                        io.BytesIO(raw)
                    )

                )

            elif suffix in [

                ".html",
                ".htm"

            ]:

                result[
                    "documentation"
                ].append(

                    read_html(
                        decoded
                    )

                )

            elif suffix==".py":

                result[
                    "code"
                ].append(decoded)

            elif suffix==".ipynb":

                result[
                    "notebooks"
                ].append(

                    read_notebook(
                        decoded
                    )

                )

            elif suffix==".csv":

                result[
                    "datasets"
                ].append(

                    summarize_csv(
                        decoded
                    )

                )

            elif suffix==".sql":

                result[
                    "database"
                ].append(decoded)

            elif suffix==".md":

                result[
                    "documentation"
                ].append(decoded)

        except:

            pass

    return result


# --------------------------------
# CONTEXT
# --------------------------------

def build_context(data):

    return f"""

DOCUMENTATION

{' '.join(data['documentation'])[:12000]}

NOTEBOOKS

{' '.join(data['notebooks'])[:10000]}

CODE

{' '.join(data['code'])[:15000]}

DATABASE

{' '.join(data['database'])[:5000]}

DATASETS

{data['datasets']}

"""


# --------------------------------
# OPENAI
# --------------------------------

def evaluate_submission(prompt):

    response=client.chat.completions.create(

        model="gpt-4.1",

        temperature=0,

        response_format={

            "type":"json_object"

        },

        messages=[

        {

        "role":"system",

        "content":"""

EXTREMELY STRICT evaluator.

Maximum displayed score=75.

Exceptional:
70-75

Strong:
65-69

Average:
45-64

Weak:
Below 45

Deduct aggressively:

- hardcoded logic
- duplicate code
- weak validation
- weak modularity
- weak architecture
- incomplete implementation

Only evidence earns marks.

Return JSON ONLY.

"""

        },

        {

        "role":"user",

        "content":prompt

        }

        ]

    )

    return response.choices[0].message.content


# --------------------------------
# JSON
# --------------------------------

def parse_json(raw):

    try:

        return json.loads(raw)

    except:

        return {

            "scores":{},
            "strengths":[],
            "improvements":[]

        }


# --------------------------------
# UI
# --------------------------------

problem=st.file_uploader(

"Problem Statement",

["pdf","docx"]

)

rubric=st.file_uploader(

"Rubric",

["xlsx"]

)

submissions=st.file_uploader(

"Participant ZIP Files",

type=["zip"],

accept_multiple_files=True

)

custom_prompt=st.text_area(

"Strict Evaluation Instructions"

)


# --------------------------------
# RUN
# --------------------------------

if st.button("Evaluate"):

    rubric_df=pd.read_excel(
        rubric
    )

    rubric_total=int(

        rubric_df[
            "Max Score"
        ].sum()

    )

    scale_factor=75/rubric_total

    rubric_text=rubric_to_text(
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

    def process(zip_obj):

        parsed=parse_submission(

            zip_obj.read()

        )

        context=build_context(
            parsed
        )

        prompt=f"""

PROBLEM

{problem_text}

RUBRIC

{rubric_text}

STRICT RULES

{custom_prompt}

SUBMISSION

{context}

Return rubric names EXACTLY.

No evidence=no marks.

"""

        raw=evaluate_submission(
            prompt
        )

        result=parse_json(
            raw
        )

        row={

            "Participant":
            zip_obj.name

        }

        total=0

        for _,r in rubric_df.iterrows():

            criterion=r[
                "Criterion"
            ]

            max_score=int(
                r["Max Score"]
            )

            raw_score=float(

                result[
                    "scores"
                ].get(

                    criterion,

                    0

                )

            )

            adjusted=int(

                round(

                    raw_score*

                    scale_factor

                )

            )

            adjusted=max(

                0,

                min(

                    adjusted,

                    max_score

                )

            )

            row[
                f"{criterion} ({max_score})"
            ]=adjusted

            total+=adjusted

        row[
            "Total"
        ]=total

        row[
            "Strengths"
        ]="; ".join(

            result.get(
                "strengths",
                []
            )

        )

        row[
            "Improvements"
        ]="; ".join(

            result.get(
                "improvements",
                []
            )

        )

        return row

    with ThreadPoolExecutor(

        max_workers=4

    ) as executor:

        results=list(

            executor.map(

                process,

                submissions

            )

        )

    output=pd.DataFrame(
        results
    )

    st.dataframe(

        output,

        use_container_width=True

    )

    excel=io.BytesIO()

    with pd.ExcelWriter(

        excel,

        engine="xlsxwriter"

    ) as writer:

        output.to_excel(

            writer,

            index=False

        )

    st.download_button(

        "📥 Download Excel",

        excel.getvalue(),

        "evaluation_report.xlsx"

    )
