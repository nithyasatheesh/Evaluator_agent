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

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="AI Case Study Evaluator",
    layout="wide"
)

st.title("📊 AI Case Study Evaluator")

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# =====================================
# READERS
# =====================================

def read_pdf(file):

    try:

        reader = PdfReader(file)

        pages=[]

        for page in reader.pages:

            text=page.extract_text()

            if text:

                pages.append(text)

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


# =====================================
# RUBRIC
# =====================================

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


# =====================================
# PARSER
# =====================================

def parse_submission(zip_bytes):

    result={

        "docs":[],

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

            text=raw.decode(

                errors="ignore"

            )

            if suffix==".pdf":

                result["docs"].append(

                    read_pdf(
                        io.BytesIO(raw)
                    )

                )

            elif suffix==".docx":

                result["docs"].append(

                    read_docx(
                        io.BytesIO(raw)
                    )

                )

            elif suffix in [

                ".html",
                ".htm"

            ]:

                result["docs"].append(

                    read_html(text)

                )

            elif suffix==".py":

                result["code"].append(
                    text
                )

            elif suffix==".ipynb":

                result["notebooks"].append(

                    read_notebook(
                        text
                    )

                )

            elif suffix==".csv":

                result["datasets"].append(

                    summarize_csv(
                        text
                    )

                )

            elif suffix==".sql":

                result["database"].append(
                    text
                )

            elif suffix==".md":

                result["docs"].append(
                    text
                )

            elif suffix in [

                ".png",
                ".jpg",
                ".jpeg"

            ]:

                result["images"].append(
                    file
                )

        except:

            pass

    return result


# =====================================
# CONTEXT
# =====================================

def build_context(parsed):

    return f"""

DOCUMENTATION

{' '.join(parsed['docs'])[:12000]}

NOTEBOOKS

{' '.join(parsed['notebooks'])[:8000]}

CODE

{' '.join(parsed['code'])[:15000]}

DATABASE

{' '.join(parsed['database'])[:5000]}

DATASETS

{parsed['datasets']}

IMAGES

{parsed['images']}

"""


# =====================================
# SCORE NORMALIZATION
# =====================================

def normalize_score(

    total,

    context

):

    size=len(context)

    multiplier=1

    if size<4000:

        multiplier=.60

    elif size<8000:

        multiplier=.72

    elif size<12000:

        multiplier=.82

    else:

        multiplier=.92

    adjusted=total*multiplier

    if adjusted>75:

        adjusted=75

    elif adjusted>=65:

        adjusted=min(
            adjusted,
            69
        )

    elif adjusted>=45:

        adjusted=min(
            adjusted,
            60
        )

    adjusted=max(

        adjusted,

        35

    )

    return round(

        adjusted,

        2

    )


# =====================================
# OPENAI
# =====================================

def evaluate(prompt):

    response=client.chat.completions.create(

        model="gpt-4.1",

        temperature=0,

        messages=[

        {

        "role":"system",

        "content":"""

You are STRICT.

Maximum excellent score:

75

Prefer scores:

45-60

Exceptional:

70-75

Evaluate ONLY evidence.

Folder count != quality.

Project size != quality.

Boilerplate != quality.

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


# =====================================
# SAFE JSON
# =====================================

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
        "overall_feedback":""

    }


# =====================================
# UI
# =====================================

problem=st.file_uploader(

"Problem",

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

"Evaluation Policy"

)


# =====================================
# RUN
# =====================================

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

EVALUATION POLICY

{custom_prompt}

Maximum excellent score:

75

Prefer:

45-60

Only exceptional:

70-75

PROBLEM

{problem_text}

RUBRIC

{rubric_text}

SUBMISSION

{context}

Return JSON:

{{
"scores":{{}},
"strengths":[],
"improvements":[],
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

            criterion=r["Criterion"]

            score=float(

                result.get(

                    "scores",

                    {}

                ).get(

                    criterion,

                    0

                )

            )

            row[criterion]=score

            total+=score

        final_score=normalize_score(

            total,

            context

        )

        row["Final Score"]=final_score

        row["Strengths"]="; ".join(

            result.get(

                "strengths",

                []

            )

        )

        row["Areas Of Improvement"]="; ".join(

            result.get(

                "improvements",

                []

            )

        )

        row["Overall Feedback"]=result.get(

            "overall_feedback",

            ""

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
