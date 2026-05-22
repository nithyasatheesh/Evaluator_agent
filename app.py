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


# ==================================
# CONFIG
# ==================================

st.set_page_config(
    page_title="AI Case Study Evaluator",
    layout="wide"
)

st.title("📊 AI Case Study Evaluator")

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)


# ==================================
# READERS
# ==================================

def read_pdf(file):

    try:

        reader = PdfReader(file)

        text=[]

        for page in reader.pages:

            t=page.extract_text()

            if t:

                text.append(t)

        return "\n".join(text)

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

    for tag in soup(

        ["script","style"]

    ):

        tag.decompose()

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


# ==================================
# RUBRIC
# ==================================

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


# ==================================
# ZIP PARSER
# ==================================

def parse_submission(zip_bytes):

    parsed={

        "docs":[],

        "code":[],

        "notebooks":[],

        "datasets":[],

        "database":[],

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

            text=raw.decode(
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

                    read_html(text)

                )

            elif suffix==".py":

                parsed["code"].append(
                    text
                )

            elif suffix==".ipynb":

                parsed["notebooks"].append(

                    read_notebook(text)

                )

            elif suffix==".csv":

                parsed["datasets"].append(

                    summarize_csv(text)

                )

            elif suffix==".sql":

                parsed["database"].append(
                    text
                )

            elif suffix==".md":

                parsed["docs"].append(
                    text
                )

            elif suffix in [

                ".png",
                ".jpg",
                ".jpeg"

            ]:

                parsed["images"].append(
                    file
                )

        except:

            pass

    return parsed


# ==================================
# BUILD CONTEXT
# ==================================

def build_context(parsed):

    return f"""

DOCUMENTATION

{' '.join(parsed['docs'])[:12000]}

NOTEBOOKS

{' '.join(parsed['notebooks'])[:9000]}

CODE

{' '.join(parsed['code'])[:15000]}

DATABASE

{' '.join(parsed['database'])[:5000]}

DATASETS

{parsed['datasets']}

IMAGES

{parsed['images']}

"""


# ==================================
# FINAL SCORE
# ==================================

def calculate_final_score(
    total,
    penalties
):

    deduction=sum(

        p.get(
            "deduction",
            0
        )

        for p in penalties

    )

    adjusted=max(

        total-deduction,

        0

    )

    final=adjusted*0.75

    return round(

        min(final,75),

        2

    )


# ==================================
# OPENAI
# ==================================

def evaluate(prompt):

    response=client.chat.completions.create(

        model="gpt-4.1",

        temperature=0,

        messages=[

        {

        "role":"system",

        "content":"""

You are STRICT.

Evaluate ONLY evidence.

Maximum excellent score:
75

Penalty system:

Weak validation=-2

Weak architecture=-2

Weak modularity=-1

Duplicate code=-1

Hardcoded values=-2

Weak documentation=-1

Weak exception handling=-1

Missing edge cases=-2

Incomplete implementation=-2

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


# ==================================
# JSON
# ==================================

def safe_json(raw):

    try:

        return json.loads(raw)

    except:

        match=re.search(

            r"\{.*\}",

            raw,

            re.DOTALL

        )

        if match:

            try:

                return json.loads(

                    match.group()

                )

            except:

                pass

    return {

        "scores":{},
        "strengths":[],
        "improvements":[],
        "penalties":[],
        "overall_feedback":""

    }


# ==================================
# UI
# ==================================

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


# ==================================
# RUN
# ==================================

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

Maximum excellent score=75

Prefer:

45-60 average

60-69 good

70-75 exceptional

PROBLEM

{problem_text}

RUBRIC

{rubric_text}

SUBMISSION

{context}

Return:

{{
"scores":{{}},
"strengths":[],
"improvements":[],
"penalties":[
{{
"reason":"",
"deduction":0
}}
],
"overall_feedback":""
}}

"""

        raw=evaluate(prompt)

        result=safe_json(raw)

        row={

            "Participant":
            file.name

        }

        total=0

        for _,r in rubric_df.iterrows():

            criterion=r[
                "Criterion"
            ]

            val=float(

                result.get(
                    "scores",
                    {}
                ).get(
                    criterion,
                    0
                )

            )

            row[
                criterion
            ]=val

            total+=val

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

            f"{p['reason']}(-{p['deduction']})"

            for p in result.get(
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

            index=False

        )

    st.download_button(

        "📥 Download Excel",

        excel.getvalue(),

        "evaluation_report.xlsx"

    )
