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


# =====================
# CONFIG
# =====================

st.set_page_config(
    page_title="AI Case Study Evaluator",
    layout="wide"
)

st.title("📊 AI Case Study Evaluator")

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)


# =====================
# READERS
# =====================

def read_pdf(file):

    try:

        reader = PdfReader(file)

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

    soup=BeautifulSoup(

        text,

        "html.parser"

    )

    for t in soup(

        ["script","style"]

    ):

        t.decompose()

    return soup.get_text(
        separator="\n"
    )


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


# =====================
# RUBRIC
# =====================

def rubric_to_text(df):

    text=[]

    for _,row in df.iterrows():

        text.append(

f"""
Criterion:
{row['Criterion']}

Max Score:
{row['Max Score']}

Description:
{row['Description']}
"""

        )

    return "\n".join(text)


# =====================
# ZIP PARSER
# =====================

def parse_submission(zip_bytes):

    parsed={

        "docs":[],

        "code":[],

        "notebooks":[],

        "database":[],

        "datasets":[]

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

                parsed["docs"].append(

                    read_pdf(
                        io.BytesIO(raw)
                    )

                )

            elif suffix==".docx":

                parsed["docs"].append(

                    read_docx(
                        io.BytesIO(raw)
                    )

                )

            elif suffix in [

                ".html",
                ".htm"

            ]:

                parsed["docs"].append(

                    read_html(decoded)

                )

            elif suffix==".py":

                parsed["code"].append(
                    decoded
                )

            elif suffix==".ipynb":

                parsed["notebooks"].append(

                    read_notebook(
                        decoded
                    )

                )

            elif suffix==".csv":

                parsed["datasets"].append(

                    summarize_csv(
                        decoded
                    )

                )

            elif suffix==".sql":

                parsed["database"].append(
                    decoded
                )

            elif suffix==".md":

                parsed["docs"].append(
                    decoded
                )

        except:

            pass

    return parsed


# =====================
# CONTEXT
# =====================

def build_context(parsed):

    return f"""

DOCUMENTATION

{' '.join(parsed['docs'])[:12000]}

NOTEBOOKS

{' '.join(parsed['notebooks'])[:10000]}

CODE

{' '.join(parsed['code'])[:15000]}

DATABASE

{' '.join(parsed['database'])[:5000]}

DATASETS

{parsed['datasets']}

"""


# =====================
# FINAL SCORE
# =====================

def calculate_final_score(
    total,
    penalties
):

    deduction=sum(

        int(

            p.get(
                "deduction",
                0
            )

        )

        for p in penalties

    )

    return min(

        max(
            total-deduction,
            0
        ),

        75

    )


# =====================
# OPENAI
# =====================

def evaluate(prompt):

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

STRICT evaluator.

Maximum excellent score=75.

Strong project:
70-75

Good:
60-69

Average:
45-60

Reduce score aggressively.

Never award full marks easily.

Return JSON ONLY.

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


# =====================
# SAFE JSON
# =====================

def safe_json(raw):

    try:

        return json.loads(raw)

    except:

        return {

            "scores":{},
            "strengths":[],
            "improvements":[],
            "penalties":[],
            "overall_feedback":""

        }


# =====================
# UI
# =====================

problem=st.file_uploader(

"Problem",

["pdf","docx"]

)

rubric=st.file_uploader(

"Rubric",

["xlsx"]

)

submissions=st.file_uploader(

"Participant ZIP",

type=["zip"],

accept_multiple_files=True

)

policy=st.text_area(

"Evaluation Policy"

)


# =====================
# RUN
# =====================

if st.button("Evaluate"):

    rubric_df=pd.read_excel(
        rubric
    )

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

    def process(file):

        parsed=parse_submission(

            file.read()

        )

        context=build_context(
            parsed
        )

        prompt=f"""

POLICY

{policy}

PROBLEM

{problem_text}

RUBRIC

{rubric_text}

SUBMISSION

{context}

IMPORTANT:

Return rubric criterion names EXACTLY.

Return:

{{
"scores":{{}},
"strengths":[],
"improvements":[],
"penalties":[],
"overall_feedback":""
}}

"""

        result=safe_json(

            evaluate(prompt)

        )

        row={

            "Participant":
            file.name

        }

        total=0

        for _,r in rubric_df.iterrows():

            criterion=r[
                "Criterion"
            ]

            max_marks=int(
                r["Max Score"]
            )

            raw=float(

                result.get(
                    "scores",
                    {}
                ).get(
                    criterion,
                    0
                )

            )

            ratio=0

            if max_marks:

                ratio=raw/max_marks

            if ratio>=0.95:

                score=max_marks-2

            elif ratio>=0.90:

                score=max_marks-3

            elif ratio>=0.80:

                score=max_marks-4

            elif ratio>=0.70:

                score=max_marks-5

            elif ratio>=0.60:

                score=max_marks-6

            else:

                score=int(
                    round(raw)
                )

            score=max(

                0,

                min(
                    score,
                    max_marks
                )

            )

            row[
                f"{criterion} ({max_marks})"
            ]=score

            total+=score

        final=calculate_final_score(

            total,

            result.get(
                "penalties",
                []
            )

        )

        row[
            "Final Score"
        ]=final

        row[
            "Strengths"
        ]="; ".join(

            result.get(
                "strengths",
                []

            )

        )

        row[
            "Areas Of Improvement"
        ]="; ".join(

            result.get(
                "improvements",
                []

            )

        )

        row[
            "Penalty Summary"
        ]="; ".join(

            [

            f"{x['reason']}(-{x['deduction']})"

            for x in result.get(
                "penalties",
                []
            )

            ]

        )

        row[
            "Overall Feedback"
        ]=result.get(

            "overall_feedback",

            ""

        )

        return row

    with ThreadPoolExecutor(
        max_workers=4
    ) as ex:

        results=list(

            ex.map(
                process,
                submissions
            )

        )

    df=pd.DataFrame(
        results
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    excel=io.BytesIO()

    with pd.ExcelWriter(

        excel,

        engine="xlsxwriter"

    ) as writer:

        df.to_excel(

            writer,

            index=False,

            sheet_name="Evaluation"

        )

    st.download_button(

        "📥 Download Excel",

        excel.getvalue(),

        "evaluation_report.xlsx"

    )
