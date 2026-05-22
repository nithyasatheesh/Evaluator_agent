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
# FILE READERS
# --------------------------------

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


def read_html(text):

    soup = BeautifulSoup(
        text,
        "html.parser"
    )

    for tag in soup(
        ["script", "style"]
    ):
        tag.decompose()

    return soup.get_text(
        separator="\n"
    )


def read_notebook(text):

    try:

        nb = nbformat.reads(
            text,
            as_version=4
        )

        output = []

        for cell in nb.cells:

            if cell.cell_type == "markdown":

                output.append(
                    "".join(cell.source)
                )

            elif cell.cell_type == "code":

                output.append(
                    "CODE:\n" +
                    "".join(cell.source)
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
Rows: {df.shape[0]}

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

    text = []

    for _, row in df.iterrows():

        text.append(

f"""
Criterion:
{row["Criterion"]}

Max Score:
{row["Max Score"]}

Description:
{row["Description"]}
"""
        )

    return "\n".join(text)


# --------------------------------
# PARSER
# --------------------------------

def parse_submission(zip_bytes):

    result = {

        "documentation": [],
        "code": [],
        "notebooks": [],
        "datasets": [],
        "database": [],
        "images": []

    }

    z = zipfile.ZipFile(
        io.BytesIO(zip_bytes)
    )

    for file in z.namelist():

        try:

            raw = z.read(file)

            suffix = Path(
                file
            ).suffix.lower()

            decoded = raw.decode(
                errors="ignore"
            )

            if suffix == ".pdf":

                result[
                    "documentation"
                ].append(

                    read_pdf(
                        io.BytesIO(raw)
                    )

                )

            elif suffix == ".docx":

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

                    read_html(decoded)

                )

            elif suffix == ".py":

                result[
                    "code"
                ].append(

                    decoded

                )

            elif suffix == ".ipynb":

                result[
                    "notebooks"
                ].append(

                    read_notebook(
                        decoded
                    )

                )

            elif suffix == ".csv":

                result[
                    "datasets"
                ].append(

                    summarize_csv(
                        decoded
                    )

                )

            elif suffix == ".sql":

                result[
                    "database"
                ].append(

                    decoded

                )

            elif suffix == ".md":

                result[
                    "documentation"
                ].append(

                    decoded

                )

            elif suffix in [

                ".png",
                ".jpg",
                ".jpeg"

            ]:

                result[
                    "images"
                ].append(file)

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

IMAGES

{data['images']}

"""


# --------------------------------
# OPENAI
# --------------------------------

def evaluate_submission(prompt):

    response = client.chat.completions.create(

        model="gpt-4.1",

        temperature=0,

        messages=[

            {

                "role":"system",

                "content":"""

You are an EXTREMELY STRICT evaluator.

Rules:

1. Never give full marks by default.

2. Missing implementation → deduct.

3. Placeholder code → deduct.

4. Hardcoded logic → deduct.

5. Missing validations → deduct.

6. Missing edge cases → deduct.

7. Weak architecture → deduct.

8. Weak modularity → deduct.

9. Weak documentation → deduct.

10. Weak frontend/backend integration → deduct.

11. Missing API handling → deduct.

12. Missing testing → deduct.

13. Average projects:
50-75

14. Strong projects:
75-88

15. Exceptional:
90+

16. Full marks only if evidence exists.

17. Different submissions should not receive identical scores unless truly identical.

Return ONLY JSON.

Format:

{
"scores":{
},
"strengths":[
],
"improvements":[
]
}

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


# --------------------------------
# SAFE JSON
# --------------------------------

def parse_json(raw):

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


# --------------------------------
# UI
# --------------------------------

problem = st.file_uploader(
    "Problem Statement",
    ["pdf","docx"]
)

rubric = st.file_uploader(
    "Rubric",
    ["xlsx"]
)

submissions = st.file_uploader(

    "Participant ZIP Files",

    type=["zip"],

    accept_multiple_files=True

)

custom_prompt = st.text_area(

"Strict Evaluation Instructions",

placeholder="""
Example:

Be extremely strict.

Deduct for:

- hardcoded values
- missing validations
- duplicated code
- weak architecture
- missing edge cases
- poor modularity

Only exceptional submissions get 90+
"""

)


# --------------------------------
# RUN
# --------------------------------

if st.button("Evaluate"):

    rubric_df = pd.read_excel(
        rubric
    )

    rubric_text = rubric_to_text(
        rubric_df
    )

    if problem.name.endswith(".pdf"):

        problem_text = read_pdf(
            problem
        )

    else:

        problem_text = read_docx(
            problem
        )

    def process(zip_obj):

        parsed = parse_submission(
            zip_obj.read()
        )

        context = build_context(
            parsed
        )

        prompt = f"""

PROBLEM:

{problem_text}

RUBRIC:

{rubric_text}

STRICT RULES:

{custom_prompt}

SUBMISSION:

{context}

Evaluate STRICTLY.

Evidence required.

No evidence = no score.

Return JSON only.

"""

        raw = evaluate_submission(
            prompt
        )

        parsed_json = parse_json(
            raw
        )

        row = {

            "Participant":
            zip_obj.name

        }

        total = 0

        for _, r in rubric_df.iterrows():

            criterion = r[
                "Criterion"
            ]

            score = float(

                parsed_json[
                    "scores"
                ].get(

                    criterion,

                    0

                )

            )

            row[
                criterion
            ] = score

            total += score

        row[
            "Total"
        ] = total

        row[
            "Strengths"
        ] = "; ".join(

            parsed_json.get(
                "strengths",
                []
            )

        )

        row[
            "Improvements"
        ] = "; ".join(

            parsed_json.get(
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

            results = list(

                executor.map(
                    process,
                    submissions
                )

            )

    output = pd.DataFrame(
        results
    )

    st.success("Done")

    st.dataframe(
        output,
        use_container_width=True
    )

    excel = io.BytesIO()

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
