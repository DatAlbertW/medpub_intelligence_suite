"""
MedPub Intelligence Suite
=========================
A POC demonstrating AI-powered publication planning for medical communications.
Built with Streamlit + Groq (Llama 3) for ContentEd Med pitch.

Pipeline:
  1. Upload PubMed CSV or PDF abstracts
  2. AI screens abstracts (include/exclude with reasoning)
  3. Evidence gap analysis + publication plan generation
"""

import streamlit as st
import pandas as pd
import json
import re
import time
import io
from groq import Groq
import fitz  # PyMuPDF

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="MedPub Intelligence Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background-color: #0D1B2A;
    color: #E8EDF2;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0A1520 !important;
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] * {
    color: #C9D6E3 !important;
}

/* ── Display font ── */
h1, h2 {
    font-family: 'DM Serif Display', serif !important;
    color: #E8EDF2 !important;
}
h3, h4 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #C9D6E3 !important;
}

/* ── Cards ── */
.card {
    background: #112236;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 4px solid #D4A843;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-include  { background: #0D3B2E; color: #34D399; border: 1px solid #34D399; }
.badge-exclude  { background: #3B0D0D; color: #F87171; border: 1px solid #F87171; }
.badge-maybe    { background: #3B2E0D; color: #FBBF24; border: 1px solid #FBBF24; }

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: #112236;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #D4A843;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #7A9BBF;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Publication plan table ── */
.pub-card {
    background: #0D1B2A;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
}
.pub-title {
    font-weight: 600;
    color: #E8EDF2;
    font-size: 1rem;
    margin-bottom: 4px;
}
.pub-meta {
    font-size: 0.82rem;
    color: #7A9BBF;
}
.pub-meta span {
    margin-right: 16px;
}

/* ── Gap item ── */
.gap-item {
    background: #0D1B2A;
    border-left: 3px solid #D4A843;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    border-radius: 0 8px 8px 0;
}

/* ── Step indicator ── */
.step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    font-size: 0.88rem;
}
.step-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-active  .step-num { background: #D4A843; color: #0A1520; }
.step-done    .step-num { background: #34D399; color: #0A1520; }
.step-pending .step-num { background: #1E3A5F; color: #7A9BBF; }
.step-active  { color: #E8EDF2; font-weight: 600; }
.step-pending { color: #7A9BBF; }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, #1E3A5F 0%, transparent 100%);
    margin: 1.5rem 0;
}

/* ── Streamlit overrides ── */
[data-testid="stButton"] > button {
    background: #D4A843 !important;
    color: #0A1520 !important;
    border: none !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
}
[data-testid="stButton"] > button:hover {
    background: #E8C060 !important;
}

div[data-testid="stFileUploader"] {
    background: #112236 !important;
    border: 1px dashed #1E3A5F !important;
    border-radius: 10px !important;
}

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #112236 !important;
    border: 1px solid #1E3A5F !important;
    color: #E8EDF2 !important;
    border-radius: 8px !important;
}

.stAlert { border-radius: 10px !important; }
div[data-testid="stExpander"] {
    background: #112236 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 10px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #D4A843 !important; }

/* ── Progress ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #D4A843, #E8C060) !important;
}

/* Info/success/warning boxes */
.stInfo    { background: #0D2438 !important; border-color: #1E3A5F !important; }
.stSuccess { background: #0D2E22 !important; border-color: #34D399 !important; }
.stWarning { background: #2E1F0D !important; border-color: #D4A843 !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.4rem;
                    color: #E8EDF2; line-height: 1.2;'>
            MedPub<br>
            <span style='color: #D4A843;'>Intelligence</span><br>
            Suite
        </div>
        <div style='font-size: 0.72rem; color: #7A9BBF; margin-top: 6px;
                    text-transform: uppercase; letter-spacing: 0.1em;'>
            AI-Powered Publication Planning
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Configuration")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )

    indication = st.text_input(
        "Therapeutic Area / Indication",
        placeholder="e.g. atezolizumab in NSCLC",
        help="Used to focus the AI analysis on relevant clinical context",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Step progress indicator ──
    st.markdown("#### Progress")

    # Determine current step from session state
    step = st.session_state.get("step", 1)

    def _step_html(num, label, current):
        if num < current:
            cls = "step-done";    icon = "✓"
        elif num == current:
            cls = "step-active";  icon = str(num)
        else:
            cls = "step-pending"; icon = str(num)
        return f"""
        <div class="step {cls}">
            <div class="step-num">{icon}</div>
            <div>{label}</div>
        </div>"""

    st.markdown(
        _step_html(1, "Upload Documents", step) +
        _step_html(2, "AI Screening", step) +
        _step_html(3, "Evidence Gaps & Plan", step),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 0.72rem; color: #3A6B9A; line-height: 1.6;'>
        <strong style='color: #5A8BAA;'>In production this connects to:</strong><br>
        PubMed API · Embase · Cochrane<br>
        Client document libraries · CTMS<br>
        Journal impact factor databases
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def get_groq_client(key: str) -> Groq:
    return Groq(api_key=key)


def extract_text_from_pdf(uploaded_file) -> list[dict]:
    """Extract title + abstract from each page of a PDF."""
    docs = []
    pdf_bytes = uploaded_file.read()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(pdf):
        text = page.get_text()
        # Naively use first 80 chars as 'title' and rest as abstract
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        title    = lines[0] if lines else f"Document page {i+1}"
        abstract = " ".join(lines[1:]) if len(lines) > 1 else text
        docs.append({"title": title, "abstract": abstract[:2000]})
    return docs


def parse_pubmed_csv(uploaded_file) -> list[dict]:
    """Parse a PubMed CSV export into a list of {title, abstract} dicts."""
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # PubMed CSV column names vary slightly — handle common variants
    title_col    = next((c for c in df.columns if "title"    in c), None)
    abstract_col = next((c for c in df.columns if "abstract" in c), None)
    year_col     = next((c for c in df.columns if "year"     in c or "publication year" in c), None)
    journal_col  = next((c for c in df.columns if "journal"  in c), None)

    records = []
    for _, row in df.iterrows():
        records.append({
            "title":    str(row[title_col])    if title_col    else "N/A",
            "abstract": str(row[abstract_col]) if abstract_col else "",
            "year":     str(row[year_col])     if year_col     else "",
            "journal":  str(row[journal_col])  if journal_col  else "",
        })
    return records


def screen_abstracts(client: Groq, documents: list[dict], indication: str) -> list[dict]:
    """
    For each document, ask the LLM to include/exclude it and provide reasoning.
    Returns original docs enriched with 'decision', 'confidence', 'reasoning'.
    """
    results = []
    progress = st.progress(0, text="Screening abstracts…")

    for i, doc in enumerate(documents):
        prompt = f"""You are a systematic review expert screening abstracts for a publication plan.

Therapeutic area / indication: {indication or "general medicine"}

Abstract to screen:
Title: {doc['title']}
Text: {doc['abstract'][:1500]}

Decide: INCLUDE, EXCLUDE, or MAYBE.
Return ONLY valid JSON (no markdown, no explanation outside JSON):
{{
  "decision": "INCLUDE" | "EXCLUDE" | "MAYBE",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence",
  "study_type": "RCT | Observational | Review | Meta-analysis | Case report | Other",
  "key_endpoint": "primary endpoint or finding in 8 words max"
}}"""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a systematic review expert. You ALWAYS respond with valid JSON only. No markdown fences, no explanations, no text before or after the JSON object."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()
            # Aggressively strip any markdown fences or surrounding text
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            # Extract just the JSON object if there's surrounding text
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)
            parsed = json.loads(raw)
        except Exception as e:
            parsed = {
                "decision": "MAYBE",
                "confidence": 0.5,
                "reasoning": f"Parsing error: {str(e)[:80]}",
                "study_type": "Unknown",
                "key_endpoint": "N/A",
            }

        results.append({**doc, **parsed})
        progress.progress((i + 1) / len(documents),
                          text=f"Screened {i+1} / {len(documents)} abstracts…")
        time.sleep(0.1)  # avoid rate limiting

    progress.empty()
    return results


def analyse_gaps_and_plan(client: Groq, screened: list[dict], indication: str) -> dict:
    """
    Send included abstracts to the LLM to identify evidence gaps
    and generate a structured publication plan.
    """
    included = [d for d in screened if d["decision"] in ("INCLUDE", "MAYBE")]
    summary_block = "\n".join(
        f"- [{d.get('study_type','?')}] {d['title']} | Endpoint: {d.get('key_endpoint','?')}"
        for d in included[:30]  # cap tokens
    )

    prompt = f"""You are a senior medical publications strategist.

Therapeutic area: {indication or "general medicine"}

Included evidence ({len(included)} studies):
{summary_block}

Your task:
1. Identify the top 4-5 evidence GAPS (topics with insufficient or missing coverage).
2. Propose a publication plan with 4-6 specific manuscripts.

Return ONLY valid JSON:
{{
  "evidence_gaps": [
    {{
      "gap": "short title of the gap",
      "description": "1-2 sentences explaining why this is a gap",
      "priority": "High | Medium | Low"
    }}
  ],
  "publication_plan": [
    {{
      "title": "Proposed manuscript title",
      "type": "Original Research | Review | Meta-analysis | Case Series | Opinion",
      "journal_tier": "Tier 1 (IF>10) | Tier 2 (IF 5-10) | Tier 3 (IF<5)",
      "addresses_gap": "which gap this addresses",
      "timeline_months": 6,
      "rationale": "one sentence"
    }}
  ],
  "strategic_summary": "2-3 sentence executive summary of the publication landscape"
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a senior medical publications strategist. You ALWAYS respond with valid JSON only. No markdown fences, no explanations, no text before or after the JSON object."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='margin-bottom: 0;'>MedPub Intelligence Suite</h1>
<p style='color: #7A9BBF; font-size: 0.95rem; margin-top: 4px;'>
    From evidence upload to publication strategy — powered by AI
</p>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD
# ════════════════════════════════════════════════════════════════════

st.markdown("### Step 1 — Upload Documents")
st.markdown("""
<div class='card'>
    <div style='color: #7A9BBF; font-size: 0.9rem;'>
        Upload a <strong style='color:#E8EDF2;'>PubMed CSV export</strong>
        or one / multiple <strong style='color:#E8EDF2;'>PDF abstracts</strong>.
        The pipeline will screen each document for relevance, identify evidence gaps,
        and generate a tailored publication plan.
    </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### 📄 PubMed CSV")
    st.caption("Export from PubMed → Send to → CSV")
    csv_file = st.file_uploader("", type=["csv"], key="csv_upload",
                                label_visibility="collapsed")

with col_right:
    st.markdown("##### 📑 PDF Abstract(s)")
    st.caption("One or multiple PDF files")
    pdf_files = st.file_uploader("", type=["pdf"], key="pdf_upload",
                                 accept_multiple_files=True,
                                 label_visibility="collapsed")

# Collect documents from uploads
documents = []
if csv_file:
    try:
        documents = parse_pubmed_csv(csv_file)
        st.success(f"✅ Loaded **{len(documents)}** records from CSV.")
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")

if pdf_files:
    for pf in pdf_files:
        extracted = extract_text_from_pdf(pf)
        documents.extend(extracted)
    st.success(f"✅ Extracted **{len(documents)}** pages from PDF(s).")

# ── Demo mode ──────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("**No file ready? Run a demo with sample data:**")
if st.button("▶ Load Sample Dataset (NSCLC / Immunotherapy)"):
    documents = [
        {"title": "Pembrolizumab vs chemotherapy in PD-L1+ NSCLC: 5-year OS update",
         "abstract": "Phase III RCT. Pembrolizumab demonstrated sustained OS benefit over platinum-doublet chemotherapy in untreated PD-L1 ≥50% NSCLC. Median OS 26.3 vs 13.4 months (HR 0.62). Grade ≥3 AEs lower in immunotherapy arm.",
         "year": "2023", "journal": "Journal of Clinical Oncology"},
        {"title": "Atezolizumab plus bevacizumab in NSCLC: real-world effectiveness",
         "abstract": "Retrospective cohort. 312 patients. Real-world ORR 38%, mPFS 7.2 months. Outcomes consistent with IMpower150 trial. Liver metastases subgroup showed pronounced benefit.",
         "year": "2023", "journal": "Lung Cancer"},
        {"title": "Quality of life outcomes with PD-1 inhibitors in elderly NSCLC patients",
         "abstract": "Pooled analysis of 4 RCTs. Patients ≥75 years (n=284). PRO data limited. EORTC QLQ-C30 global health scores improved vs chemotherapy but data sparse beyond 12 months. Evidence gap identified.",
         "year": "2022", "journal": "ESMO Open"},
        {"title": "CNS metastases management in NSCLC: a systematic review",
         "abstract": "28 studies included. Immunotherapy shows intracranial activity but head-to-head data lacking. Stereotactic radiosurgery combinations underexplored. Major gap in prospective CNS-specific endpoints.",
         "year": "2023", "journal": "Neuro-Oncology"},
        {"title": "Biomarker-driven selection beyond PD-L1 in NSCLC immunotherapy",
         "abstract": "Review of TMB, MSI, STK11, KEAP1 mutations as predictors of IO response. TMB utility controversial. Combined biomarker panels show promise but require prospective validation.",
         "year": "2024", "journal": "Nature Reviews Clinical Oncology"},
        {"title": "Immunotherapy rechallenge after progression: outcomes in NSCLC",
         "abstract": "Retrospective analysis. n=89. Rechallenge ORR 12%, DCR 41%. Predictors of benefit unclear. No prospective data available. Significant unmet clinical need.",
         "year": "2022", "journal": "Clinical Lung Cancer"},
        {"title": "Patient-reported fatigue during immunotherapy: NSCLC longitudinal study",
         "abstract": "Prospective observational. Fatigue most common patient-reported symptom. Severity peaks at week 6-9. No standardized management protocol exists. PRO collection inconsistent across sites.",
         "year": "2023", "journal": "Supportive Care in Cancer"},
        {"title": "Cost-effectiveness analysis of first-line pembrolizumab in NSCLC",
         "abstract": "Health economic model. ICER $85,000/QALY in PD-L1 high population. Value highly sensitive to OS tail assumptions. Real-world data needed to validate long-term projections.",
         "year": "2023", "journal": "Value in Health"},
    ]
    st.session_state["documents"] = documents
    st.success(f"✅ Sample dataset loaded — **{len(documents)} abstracts** ready.")
    st.session_state["step"] = 2
    st.rerun()

# Persist documents
if documents:
    st.session_state["documents"] = documents


# ════════════════════════════════════════════════════════════════════
#  STEP 2 — AI SCREENING
# ════════════════════════════════════════════════════════════════════

if st.session_state.get("documents"):
    documents = st.session_state["documents"]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Step 2 — AI Abstract Screening")

    st.markdown(f"""
    <div class='card card-accent'>
        <div style='color: #C9D6E3; font-size: 0.9rem;'>
            <strong>{len(documents)}</strong> documents loaded.
            The AI will assess each for clinical relevance to
            <strong style='color:#D4A843;'>{indication or "the selected indication"}</strong>,
            classify the study type, extract the key endpoint, and recommend
            include / exclude / maybe with a confidence score — mirroring
            tools like <em>Rayyan</em>, built in-house.
        </div>
    </div>
    """, unsafe_allow_html=True)

    run_screening = st.button("🤖 Run AI Screening")

    if run_screening:
        if not groq_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        else:
            client = get_groq_client(groq_key)
            with st.spinner("Screening abstracts with Llama 3…"):
                screened = screen_abstracts(client, documents, indication)
            st.session_state["screened"] = screened
            st.session_state["step"] = 3
            st.rerun()

    # ── Display screening results ─────────────────────────────────
    if st.session_state.get("screened"):
        screened = st.session_state["screened"]

        n_inc = sum(1 for d in screened if d["decision"] == "INCLUDE")
        n_exc = sum(1 for d in screened if d["decision"] == "EXCLUDE")
        n_may = sum(1 for d in screened if d["decision"] == "MAYBE")

        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-card'>
                <div class='metric-value' style='color:#34D399'>{n_inc}</div>
                <div class='metric-label'>Included</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value' style='color:#FBBF24'>{n_may}</div>
                <div class='metric-label'>Maybe</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value' style='color:#F87171'>{n_exc}</div>
                <div class='metric-label'>Excluded</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{len(screened)}</div>
                <div class='metric-label'>Total</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Badge map
        badge_map = {
            "INCLUDE": "badge-include",
            "EXCLUDE": "badge-exclude",
            "MAYBE":   "badge-maybe",
        }

        # Sort: INCLUDE first, then MAYBE, then EXCLUDE
        order = {"INCLUDE": 0, "MAYBE": 1, "EXCLUDE": 2}
        screened_sorted = sorted(screened, key=lambda x: order.get(x["decision"], 3))

        for doc in screened_sorted:
            dec   = doc.get("decision", "MAYBE")
            badge = badge_map.get(dec, "badge-maybe")
            conf  = doc.get("confidence", 0)
            conf_pct = int(conf * 100)

            st.markdown(f"""
            <div class='card' style='margin-bottom:0.6rem;'>
                <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                    <div style='flex:1; padding-right:1rem;'>
                        <div style='font-weight:600; color:#E8EDF2; margin-bottom:4px;'>
                            {doc['title'][:120]}
                        </div>
                        <div style='font-size:0.8rem; color:#7A9BBF; margin-bottom:8px;'>
                            {doc.get('study_type','?')} &nbsp;·&nbsp;
                            {doc.get('year','')} &nbsp;·&nbsp;
                            {doc.get('journal','')}
                        </div>
                        <div style='font-size:0.82rem; color:#C9D6E3;'>
                            🎯 <strong>Key endpoint:</strong> {doc.get('key_endpoint','N/A')}
                        </div>
                        <div style='font-size:0.82rem; color:#7A9BBF; margin-top:4px;'>
                            {doc.get('reasoning','')}
                        </div>
                    </div>
                    <div style='text-align:center; flex-shrink:0;'>
                        <span class='badge {badge}'>{dec}</span>
                        <div style='font-size:0.75rem; color:#7A9BBF; margin-top:6px;'>
                            {conf_pct}% conf.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Export screened results ───────────────────────────────
        df_export = pd.DataFrame([{
            "Title":       d["title"],
            "Decision":    d["decision"],
            "Confidence":  d.get("confidence", ""),
            "Study Type":  d.get("study_type", ""),
            "Key Endpoint":d.get("key_endpoint", ""),
            "Reasoning":   d.get("reasoning", ""),
            "Year":        d.get("year", ""),
            "Journal":     d.get("journal", ""),
        } for d in screened])

        st.download_button(
            label="⬇ Download Screening Results (CSV)",
            data=df_export.to_csv(index=False).encode(),
            file_name="screening_results.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════════
#  STEP 3 — EVIDENCE GAPS & PUBLICATION PLAN
# ════════════════════════════════════════════════════════════════════

if st.session_state.get("screened"):
    screened = st.session_state["screened"]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Step 3 — Evidence Gaps & Publication Plan")

    st.markdown("""
    <div class='card card-accent'>
        <div style='color: #C9D6E3; font-size: 0.9rem;'>
            Based on the screened literature, the AI will now identify
            <strong>evidence gaps</strong> — areas with insufficient published data —
            and generate a prioritised <strong>publication plan</strong> with
            manuscript types, target journal tiers, and timelines.
        </div>
    </div>
    """, unsafe_allow_html=True)

    run_analysis = st.button("📊 Generate Evidence Gap Analysis & Publication Plan")

    if run_analysis:
        if not groq_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        else:
            client = get_groq_client(groq_key)
            with st.spinner("Analysing evidence landscape…"):
                analysis = analyse_gaps_and_plan(client, screened, indication)
            st.session_state["analysis"] = analysis
            st.rerun()

    if st.session_state.get("analysis"):
        analysis = st.session_state["analysis"]

        # ── Strategic summary ─────────────────────────────────────
        summary = analysis.get("strategic_summary", "")
        if summary:
            st.markdown(f"""
            <div class='card' style='border-color:#D4A843;'>
                <div style='font-size:0.75rem; color:#D4A843; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>
                    Executive Summary
                </div>
                <div style='color:#E8EDF2; font-size:0.95rem; line-height:1.6;'>
                    {summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Evidence gaps ─────────────────────────────────────────
        st.markdown("#### 🔍 Evidence Gaps Identified")
        priority_color = {"High": "#F87171", "Medium": "#FBBF24", "Low": "#34D399"}

        for gap in analysis.get("evidence_gaps", []):
            pcolor = priority_color.get(gap.get("priority", "Medium"), "#7A9BBF")
            st.markdown(f"""
            <div class='gap-item'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <strong style='color:#E8EDF2;'>{gap['gap']}</strong>
                    <span class='badge' style='border-color:{pcolor}; color:{pcolor};
                          background:transparent;'>{gap.get('priority','?')}</span>
                </div>
                <div style='color:#9AB4CC; font-size:0.85rem; margin-top:4px;'>
                    {gap['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Publication plan ──────────────────────────────────────
        st.markdown("#### 📋 Proposed Publication Plan")

        tier_color = {
            "Tier 1 (IF>10)":  "#D4A843",
            "Tier 2 (IF 5-10)": "#7A9BBF",
            "Tier 3 (IF<5)":   "#4A6B8A",
        }

        for i, pub in enumerate(analysis.get("publication_plan", []), 1):
            tier   = pub.get("journal_tier", "")
            tcolor = tier_color.get(tier, "#7A9BBF")
            months = pub.get("timeline_months", "?")

            st.markdown(f"""
            <div class='pub-card'>
                <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                    <div>
                        <div style='font-size:0.72rem; color:#7A9BBF; font-weight:700;
                                    text-transform:uppercase; margin-bottom:4px;'>
                            #{i} · {pub.get('type','')}
                        </div>
                        <div class='pub-title'>{pub['title']}</div>
                        <div style='font-size:0.82rem; color:#9AB4CC; margin-top:6px;'>
                            {pub.get('rationale','')}
                        </div>
                        <div style='font-size:0.78rem; color:#7A9BBF; margin-top:8px;'>
                            <em>Addresses:</em> {pub.get('addresses_gap','')}
                        </div>
                    </div>
                    <div style='text-align:right; flex-shrink:0; padding-left:1rem;'>
                        <div style='font-size:0.78rem; color:{tcolor}; font-weight:600;
                                    margin-bottom:6px;'>{tier}</div>
                        <div style='font-size:0.78rem; color:#7A9BBF;'>⏱ {months} months</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Export publication plan ────────────────────────────────
        df_plan = pd.DataFrame(analysis.get("publication_plan", []))
        if not df_plan.empty:
            st.download_button(
                label="⬇ Download Publication Plan (CSV)",
                data=df_plan.to_csv(index=False).encode(),
                file_name="publication_plan.csv",
                mime="text/csv",
            )

        # ── Scalability note ──────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#0A1520; border:1px dashed #1E3A5F; border-radius:10px;
                    padding:1.2rem 1.5rem;'>
            <div style='font-size:0.72rem; color:#D4A843; font-weight:700;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;'>
                🚀 Production Roadmap
            </div>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;
                        font-size:0.82rem; color:#7A9BBF;'>
                <div>→ Direct PubMed / Embase API integration</div>
                <div>→ Automatic journal IF lookup & ranking</div>
                <div>→ Multi-language abstract support</div>
                <div>→ Client-specific publication history import</div>
                <div>→ PRISMA-compliant flow diagram export</div>
                <div>→ KOL identification from author networks</div>
                <div>→ Regulatory submission-ready evidence tables</div>
                <div>→ Timeline Gantt chart with milestone alerts</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem; text-align:center; font-size:0.72rem; color:#2A4A6A;'>
    MedPub Intelligence Suite · Proof of Concept ·
    Built with Streamlit + Groq (Llama 3)
</div>
""", unsafe_allow_html=True)
