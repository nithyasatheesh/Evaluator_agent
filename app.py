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

# ------------------------
# CONFIG
# ------------------------

st.set_page_config(
    page_title="AI Case Study Evaluator",
    layout="wide"
)

st.title("📊 AI Case Study Evaluator")

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# ------------------------
# READERS
# ------------------------

def read_pdf(file):

    try:

        reader = PdfReader(file)

        pages = []

        for p in reader.pages:

            txt = p.extract_text()

            if txt:

                pages.append(txt)

        return "\n".join(pages)

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

def read_html(text):

    soup = BeautifulSoup(
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

        nb = nbformat.reads(
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

        df = pd.read_csv(
            io.StringIO(text)
        )

        return f"""
Rows:{df.shape[0]}

Columns:
{list(df.columns)}

Sample:

{df.head(3)}
"""

    except:

        return ""

# ------------------------
# RUBRIC
# ------------------------

def rubric_to_text(df):

    txt=[]

    for _,r in df.iterrows():

        txt.append(

f"""
Criterion:
{r['Criterion']}

Max Score:
{r['Max Score']}

Description:
{r['Description']}
"""
        )

    return "\n".join(txt)

# ------------------------
# ZIP PARSER
# ------------------------

def parse_submission(
    zip_bytes
):

    data={

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

                data["docs"].append(

                    read_pdf(
                        io.BytesIO(raw)
                    )

                )

            elif suffix==".docx":

                data["docs"].append(

                    read_docx(
                        io.BytesIO(raw)
                    )

                )

            elif suffix in [

                ".html",
                ".htm"

            ]:

                data["docs"].append(

                    read_html(text)

                )

            elif suffix==".py":

                data["code"].append(

                    text

                )

            elif suffix==".ipynb":

                data["notebooks"].append(

                    read_notebook(
                        text
                    )

                )

            elif suffix==".csv":

                data["datasets"].append(

                    summarize_csv(
                        text
                    )

                )

            elif suffix==".sql":

                data["database"].append(

                    text

                )

            elif suffix==".md":

                data["docs"].append(

                    text

                )

            elif suffix in [

                ".png",
                ".jpg",
                ".jpeg"

            ]:

                data["images"].append(
                    file
                )

        except:

            pass

    return data

# ------------------------
# CONTEXT
# ------------------------

def build_context(data):

    return f"""

DOCUMENTATION

{' '.join(data['docs'])[:12000]}

NOTEBOOKS

{' '.join(data['notebooks'])[:10000]}

CODE

{' '.join(data['code'])[:12000]}

DATABASE

{' '.join(data['database'])[:5000]}

DATASETS

{data['datasets']}

IMAGES

{data['images']}

"""

# ------------------------
# OPENAI
# ------------------------

def evaluate(prompt):

    response = client.chat.completions.create(

        model="gpt-4.1",

        temperature=0,

        messages=[

            {

                "role":"system",

                "content":"""

You are a strict evaluator.

Evaluate ENTIRE submission.

Do not score files separately.

Return ONLY JSON.

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

# ------------------------
# PARSE JSON
# ------------------------

def safe_json(raw):

    try:

        return json.loads(raw)

    except:

        match = re.search(

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

        "improvements":[]

    }

# ------------------------
# UI
# ------------------------

problem=st.file_uploader(

"Upload Problem",

["pdf","docx"]

)

rubric=st.file_uploader(

"Upload Rubric",

["xlsx"]

)

submissions=st.file_uploader(

"Upload Participant ZIP Files",

type=["zip"],

accept_multiple_files=True

)

extra_prompt=st.text_area(

"Additional Instructions"

)

# ------------------------
# MAIN
# ------------------------

if st.button("Evaluate"):

    if not problem:

        st.error(
            "Upload problem"
        )

        st.stop()

    if not rubric:

        st.error(
            "Upload rubric"
        )

        st.stop()

    if not submissions:

        st.error(
            "Upload submissions"
        )

        st.stop()

    rubric_df = pd.read_excel(
        rubric
    )

    rubric_text = rubric_to_text(
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

    def process(zipfile_obj):

        parsed=parse_submission(

            zipfile_obj.read()

        )

        context=build_context(
            parsed
        )

        prompt=f"""

PROBLEM

{problem_text}

RUBRIC

{rubric_text}

EXTRA INSTRUCTIONS

{extra_prompt}

SUBMISSION

{context}

Return:

{{
"scores":{{
"criterion":score
}},
"strengths":[
""
],
"improvements":[
""
]
}}

"""

        raw=evaluate(
            prompt
        )

        result=safe_json(
            raw
        )

        row={

            "Participant":
            zipfile_obj.name

        }

        total=0

        scores=result.get(
            "scores",
            {}
        )

        for _,r in rubric_df.iterrows():

            criterion=r[
                "Criterion"
            ]

            val=float(

                scores.get(

                    criterion,

                    0

                )

            )

            row[
                criterion

            ]=val

            total+=val

        row[
            "Total"
        ]=round(
            total,
            2
        )

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

    with st.spinner(
        "Evaluating..."
    ):

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

    st.success(
        "Done"
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

            index=False,

            sheet_name="Scores"

        )

    st.download_button(

        "📥 Download Excel",

        excel.getvalue(),

        file_name=
        "evaluation_report.xlsx",

        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    )
