import io
import json
import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nbformat
import pandas as pd
import streamlit as st
import yaml
from bs4 import BeautifulSoup
from docx import Document
from openai import OpenAI, OpenAIError
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

# --------------------------------
# CONFIG
# --------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Case Study Evaluator", layout="wide")
st.title("📊 AI Case Study Evaluator")

OPENAI_MODEL = "gpt-4.1"
MAX_WORKERS = 4
MAX_CONTEXT_CHARS = {
    "documentation": 12000,
    "notebooks": 10000,
    "code": 15000,
    "database": 5000,
    "datasets": 4000,
    "structured_data": 4000,
}

SUPPORTED_SUBMISSION_TYPES = {
    ".pdf",
    ".docx",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".py",
    ".ipynb",
    ".sql",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".png",
    ".jpg",
    ".jpeg",
}

RUBRIC_COLUMN_ALIASES = {
    "criterion": ["criterion", "criteria", "evaluation criteria"],
    "max_score": ["max score", "max marks", "marks", "score", "weight"],
    "description": ["description", "evaluation parameters", "details", "evaluation description"],
}


def get_openai_client() -> Optional[OpenAI]:
    """Create an OpenAI client and show a friendly error when the key is unavailable."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key is missing. Please add OPENAI_API_KEY to Streamlit secrets.")
        return None
    return OpenAI(api_key=api_key, timeout=120, max_retries=2)


client = get_openai_client()

# --------------------------------
# FILE READERS
# --------------------------------


def decode_bytes(raw: bytes) -> str:
    """Decode text-like files without crashing on unusual encodings."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode(errors="ignore")


def read_pdf(file: io.BytesIO) -> str:
    try:
        reader = PdfReader(file)
        text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(part for part in text if part.strip())
    except (PdfReadError, OSError, ValueError, TypeError) as exc:
        logger.warning("Unable to parse PDF: %s", exc)
        return ""


def read_docx(file: io.BytesIO) -> str:
    try:
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning("Unable to parse DOCX: %s", exc)
        return ""


def read_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def read_notebook(text: str) -> str:
    try:
        nb = nbformat.reads(text, as_version=4)
    except (nbformat.reader.NotJSONError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Unable to parse notebook: %s", exc)
        return ""

    output: List[str] = []
    for cell in nb.cells:
        source = "".join(cell.get("source", ""))
        if not source.strip():
            continue
        if cell.cell_type == "markdown":
            output.append(source)
        elif cell.cell_type == "code":
            output.append(f"CODE:\n{source}")
    return "\n".join(output)


def summarize_csv(text: str) -> str:
    try:
        df = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Unable to parse CSV: %s", exc)
        return text[:2000]

    return f"""
Rows: {df.shape[0]}
Columns: {list(df.columns)}
Sample:
{df.head(5).to_string(index=False)}
"""


def summarize_structured_data(text: str, suffix: str) -> str:
    """Summarize JSON/YAML while retaining useful implementation evidence."""
    try:
        data = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
        pretty = json.dumps(data, indent=2, default=str)
        return pretty[:4000]
    except (json.JSONDecodeError, yaml.YAMLError, TypeError, ValueError) as exc:
        logger.warning("Unable to parse %s: %s", suffix, exc)
        return text[:4000]


# --------------------------------
# RUBRIC
# --------------------------------


def normalize_column_name(column: Any) -> str:
    return re.sub(r"\s+", " ", str(column).strip().lower())


def find_matching_column(columns: Iterable[Any], aliases: List[str]) -> Optional[str]:
    normalized = {normalize_column_name(col): col for col in columns}
    for alias in aliases:
        match = normalized.get(normalize_column_name(alias))
        if match is not None:
            return match
    return None


def normalize_rubric(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Map common rubric column variants to canonical names."""
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    mapping: Dict[str, str] = {}
    missing: List[str] = []
    for canonical_name, aliases in RUBRIC_COLUMN_ALIASES.items():
        column = find_matching_column(df.columns, aliases)
        if column is None:
            missing.append(" / ".join(aliases))
        else:
            mapping[column] = canonical_name

    if missing:
        return None, missing

    normalized = df.rename(columns=mapping)[["criterion", "max_score", "description"]]
    normalized = normalized.dropna(subset=["criterion", "max_score"], how="any")
    normalized["criterion"] = normalized["criterion"].astype(str).str.strip()
    normalized["description"] = normalized["description"].fillna("").astype(str).str.strip()
    normalized["max_score"] = pd.to_numeric(normalized["max_score"], errors="coerce")
    normalized = normalized.dropna(subset=["max_score"])
    return normalized, []


def rubric_to_text(df: pd.DataFrame) -> str:
    text = []
    for _, row in df.iterrows():
        text.append(
            f"""
Criterion:
{row['criterion']}

Max Score:
{row['max_score']}

Description:
{row['description']}
"""
        )
    return "\n".join(text)


# --------------------------------
# PARSER
# --------------------------------


def empty_submission_result() -> Dict[str, List[str]]:
    return {
        "documentation": [],
        "code": [],
        "notebooks": [],
        "datasets": [],
        "database": [],
        "structured_data": [],
        "images": [],
        "unsupported": [],
        "parse_warnings": [],
    }


def add_parsed_file(result: Dict[str, List[str]], filename: str, raw: bytes) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUBMISSION_TYPES:
        result["unsupported"].append(filename)
        return

    decoded = decode_bytes(raw)
    parsed_text = ""

    if suffix == ".pdf":
        parsed_text = read_pdf(io.BytesIO(raw))
        result["documentation"].append(parsed_text)
    elif suffix == ".docx":
        parsed_text = read_docx(io.BytesIO(raw))
        result["documentation"].append(parsed_text)
    elif suffix in [".html", ".htm"]:
        parsed_text = read_html(decoded)
        result["documentation"].append(parsed_text)
    elif suffix in [".md", ".txt"]:
        parsed_text = decoded
        result["documentation"].append(parsed_text)
    elif suffix == ".py":
        parsed_text = decoded
        result["code"].append(parsed_text)
    elif suffix == ".ipynb":
        parsed_text = read_notebook(decoded)
        result["notebooks"].append(parsed_text)
    elif suffix == ".csv":
        parsed_text = summarize_csv(decoded)
        result["datasets"].append(parsed_text)
    elif suffix == ".sql":
        parsed_text = decoded
        result["database"].append(parsed_text)
    elif suffix in [".json", ".yaml", ".yml"]:
        parsed_text = summarize_structured_data(decoded, suffix)
        result["structured_data"].append(parsed_text)
    elif suffix in [".png", ".jpg", ".jpeg"]:
        result["images"].append(filename)

    if suffix not in [".png", ".jpg", ".jpeg"] and not parsed_text.strip():
        result["parse_warnings"].append(f"Could not extract readable text from {filename}.")


def parse_submission(zip_bytes: bytes) -> Dict[str, List[str]]:
    result = empty_submission_result()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            for file_info in archive.infolist():
                if file_info.is_dir():
                    continue
                try:
                    add_parsed_file(result, file_info.filename, archive.read(file_info))
                except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                    logger.warning("Unable to parse %s: %s", file_info.filename, exc)
                    result["parse_warnings"].append(f"Skipped {file_info.filename}: {exc}")
    except zipfile.BadZipFile as exc:
        logger.warning("Corrupted ZIP file: %s", exc)
        result["parse_warnings"].append("The ZIP file is corrupted or not a valid ZIP archive.")
    return result


# --------------------------------
# CONTEXT
# --------------------------------


def truncate_join(items: List[str], limit: int) -> str:
    return "\n".join(item for item in items if item and item.strip())[:limit]


def build_context(data: Dict[str, List[str]]) -> str:
    return f"""
DOCUMENTATION
{truncate_join(data['documentation'], MAX_CONTEXT_CHARS['documentation'])}

NOTEBOOKS
{truncate_join(data['notebooks'], MAX_CONTEXT_CHARS['notebooks'])}

CODE
{truncate_join(data['code'], MAX_CONTEXT_CHARS['code'])}

DATABASE
{truncate_join(data['database'], MAX_CONTEXT_CHARS['database'])}

DATASETS
{truncate_join(data['datasets'], MAX_CONTEXT_CHARS['datasets'])}

STRUCTURED DATA / CONFIGURATION
{truncate_join(data['structured_data'], MAX_CONTEXT_CHARS['structured_data'])}

IMAGES
{data['images']}

PARSER WARNINGS
{data['parse_warnings']}
"""


def submission_has_evidence(data: Dict[str, List[str]]) -> bool:
    text_sections = ["documentation", "code", "notebooks", "datasets", "database", "structured_data"]
    return any(any(item.strip() for item in data[section]) for section in text_sections) or bool(data["images"])


# --------------------------------
# OPENAI
# --------------------------------


def build_evaluation_prompt(problem_text: str, rubric_text: str, custom_prompt: str, context: str) -> str:
    return f"""
PROBLEM:
{problem_text}

RUBRIC:
{rubric_text}

STRICT RULES FROM EVALUATOR:
{custom_prompt}

SUBMISSION:
{context}

Evaluate STRICTLY against only the rubric above.

Requirements:
- Follow every rubric criterion and maximum score exactly.
- Provide quantitative scores for every criterion using the criterion names exactly as written.
- Do not assume every criterion is out of 10; never exceed the listed max score.
- Deduct marks only when required evidence is missing, incomplete, incorrect, unsupported, or poorly implemented.
- Avoid identical scores across criteria unless the evidence truly supports identical performance.
- Evaluate implementation correctness, code quality, documentation, business understanding, architecture, maintainability, and testing where relevant to the rubric and assignment.
- Consider Machine Learning, Deep Learning, Generative AI, RAG, NLP, Computer Vision, Data Engineering, Python, SQL, APIs, BI, analytics, and general coding evidence as applicable.
- No evidence = no score for that criterion.

Return ONLY valid JSON in this exact shape:
{{
  "scores": {{
    "criterion name": numeric_score
  }},
  "qualitative_feedback": {{
    "language_feedback": "",
    "analysis_feedback": "",
    "clarity_feedback": "",
    "overall_feedback": ""
  }},
  "strengths": [],
  "improvements": []
}}
"""


def evaluate_submission(prompt: str) -> str:
    if client is None:
        raise RuntimeError("OpenAI client is not configured.")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
You are an extremely strict but fair assignment evaluator.
Award marks only for evidence present in the submitted files.
Use the rubric as the source of truth for all scores and maximum marks.
Return only valid JSON with scores, qualitative_feedback, strengths, and improvements.
""",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "{}"


# --------------------------------
# SAFE JSON AND SCORING
# --------------------------------


def default_evaluation(error_message: str = "") -> Dict[str, Any]:
    return {
        "scores": {},
        "qualitative_feedback": {
            "language_feedback": "",
            "analysis_feedback": "",
            "clarity_feedback": "",
            "overall_feedback": error_message,
        },
        "strengths": [],
        "improvements": [error_message] if error_message else [],
    }


def parse_json(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            logger.warning("OpenAI response did not contain JSON.")
            return default_evaluation("The model response could not be parsed as JSON.")
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse extracted JSON: %s", exc)
            return default_evaluation("The model response contained invalid JSON.")

    if not isinstance(parsed, dict):
        return default_evaluation("The model response JSON was not an object.")
    parsed.setdefault("scores", {})
    parsed.setdefault("qualitative_feedback", {})
    parsed.setdefault("strengths", [])
    parsed.setdefault("improvements", [])
    return parsed


def coerce_score(value: Any, max_score: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(score, float(max_score)))


def get_score_for_criterion(scores: Dict[str, Any], criterion: str, max_score: float) -> float:
    if criterion in scores:
        return coerce_score(scores[criterion], max_score)
    normalized_scores = {normalize_column_name(key): value for key, value in scores.items()}
    return coerce_score(normalized_scores.get(normalize_column_name(criterion), 0), max_score)


def list_to_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def feedback_to_text(feedback: Any) -> str:
    if isinstance(feedback, dict):
        return " | ".join(f"{key}: {value}" for key, value in feedback.items() if str(value).strip())
    return str(feedback or "")


# --------------------------------
# UI
# --------------------------------

problem = st.file_uploader("Problem Statement", ["pdf", "docx"])
rubric = st.file_uploader("Rubric", ["xlsx"])
submissions = st.file_uploader("Participant ZIP Files", type=["zip"], accept_multiple_files=True)
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
""",
)


# --------------------------------
# RUN
# --------------------------------


def load_problem_text(uploaded_problem: Any) -> str:
    suffix = Path(uploaded_problem.name).suffix.lower()
    if suffix == ".pdf":
        return read_pdf(io.BytesIO(uploaded_problem.getvalue()))
    if suffix == ".docx":
        return read_docx(io.BytesIO(uploaded_problem.getvalue()))
    return ""


def process_submission(zip_name: str, zip_bytes: bytes, problem_text: str, rubric_df: pd.DataFrame, rubric_text: str) -> Dict[str, Any]:
    parsed_submission = parse_submission(zip_bytes)
    row: Dict[str, Any] = {"Participant": zip_name}

    if not submission_has_evidence(parsed_submission):
        parsed_json = default_evaluation("No readable submission evidence was found. Unsupported, empty, or corrupted files may have been uploaded.")
    else:
        prompt = build_evaluation_prompt(problem_text, rubric_text, custom_prompt, build_context(parsed_submission))
        try:
            parsed_json = parse_json(evaluate_submission(prompt))
        except (OpenAIError, TimeoutError, RuntimeError, ValueError) as exc:
            logger.exception("OpenAI evaluation failed for %s", zip_name)
            parsed_json = default_evaluation(f"OpenAI evaluation failed: {exc}")

    total = 0.0
    max_total = 0.0
    scores = parsed_json.get("scores", {}) if isinstance(parsed_json.get("scores", {}), dict) else {}
    for _, rubric_row in rubric_df.iterrows():
        criterion = str(rubric_row["criterion"])
        max_score = float(rubric_row["max_score"])
        score = get_score_for_criterion(scores, criterion, max_score)
        row[criterion] = score
        total += score
        max_total += max_score

    row["Total"] = total
    row["Max Total"] = max_total
    row["Percentage"] = round((total / max_total) * 100, 2) if max_total else 0.0
    row["Qualitative Feedback"] = feedback_to_text(parsed_json.get("qualitative_feedback"))
    row["Strengths"] = list_to_text(parsed_json.get("strengths"))
    row["Improvements"] = list_to_text(parsed_json.get("improvements"))
    row["Parser Warnings"] = list_to_text(parsed_submission.get("parse_warnings", []))
    return row


if st.button("Evaluate"):
    if client is None:
        st.stop()
    if rubric is None:
        st.error("Please upload a rubric Excel file before evaluating.")
        st.stop()
    if problem is None:
        st.error("Please upload a problem statement before evaluating.")
        st.stop()
    if not submissions:
        st.error("Please upload at least one participant ZIP file.")
        st.stop()

    try:
        raw_rubric_df = pd.read_excel(rubric)
    except (ValueError, OSError, ImportError) as exc:
        st.error(f"Could not read the rubric Excel file: {exc}")
        st.stop()

    normalized_rubric_df, missing_columns = normalize_rubric(raw_rubric_df)
    if missing_columns or normalized_rubric_df is None or normalized_rubric_df.empty:
        st.error(
            "The rubric is missing required columns. Accepted column names are: "
            "Criterion/Criteria/Evaluation Criteria, Max Score/Max Marks/Marks/Score/Weight, "
            "and Description/Evaluation Parameters/Details/Evaluation Description. "
            f"Missing: {', '.join(missing_columns) if missing_columns else 'valid rubric rows'}."
        )
        st.stop()

    problem_text = load_problem_text(problem)
    if not problem_text.strip():
        st.warning("The problem statement could not be parsed as text. Evaluation will continue using the rubric and submissions.")

    rubric_text = rubric_to_text(normalized_rubric_df)
    submission_payloads = [(uploaded.name, uploaded.getvalue()) for uploaded in submissions]
    results: List[Dict[str, Any]] = []

    progress_bar = st.progress(0)
    status = st.empty()
    status.info("Preparing evaluations...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_submission, name, payload, problem_text, normalized_rubric_df, rubric_text): name
            for name, payload in submission_payloads
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                results.append(future.result())
                status.info(f"Evaluated {completed}/{len(futures)} submissions. Latest: {name}")
            except (RuntimeError, ValueError, OSError) as exc:
                logger.exception("Unexpected failure while processing %s", name)
                results.append({"Participant": name, "Total": 0.0, "Improvements": f"Evaluation failed: {exc}"})
                status.warning(f"Skipped {name} because of an unexpected processing error.")
            progress_bar.progress(completed / len(futures))

    output = pd.DataFrame(results)
    st.success("Done")
    st.dataframe(output, use_container_width=True)

    excel = io.BytesIO()
    with pd.ExcelWriter(excel, engine="xlsxwriter") as writer:
        output.to_excel(writer, index=False, sheet_name="Scores")

    st.download_button(
        "📥 Download Excel",
        excel.getvalue(),
        file_name="evaluation_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
