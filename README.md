# MedPub Intelligence Suite 🔬

> AI-powered publication planning for med. communications.  
> Built as a POC using Streamlit + Groq (Llama 3).

---

## What it does

A 3-step pipeline that transforms a set of PubMed abstracts into a full publication strategy:

| Step | What happens |
|------|-------------|
| **1. Upload** | PubMed CSV export or PDF abstracts |
| **2. AI Screening** | Include / Exclude / Maybe with confidence scores & reasoning |
| **3. Gap Analysis + Plan** | Evidence gaps identified → publication plan with manuscript types, journal tiers, timelines |

---

## Quick start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/medpub-intelligence.git
cd medpub-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up (free, no credit card)
- Create an API key → copy it

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Use it
- Paste your Groq API key in the sidebar
- Enter a therapeutic area (e.g. `atezolizumab in NSCLC`)
- Upload a PubMed CSV **or** click **Load Sample Dataset** to demo instantly
- Run screening → then generate the publication plan

---

## How to export a PubMed CSV

1. Go to [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov)
2. Run any search
3. Click **Save** → Format: **CSV** → Create file

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| LLM | Groq API — Llama 3 70B |
| PDF parsing | PyMuPDF (fitz) |
| Data | Pandas |
| Hosting | Streamlit Community Cloud (free) |

---

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → deploy
4. Add `GROQ_API_KEY` as a secret in Streamlit Cloud settings  
   (then update `app.py` line: `groq_key = st.secrets["GROQ_API_KEY"]`)

---

## Production roadmap

- Direct PubMed / Embase API integration (no manual CSV export)
- Automatic journal impact factor lookup
- PRISMA-compliant flow diagram export
- Client publication history import
- KOL identification from author co-citation networks
- Regulatory submission-ready evidence tables
- Multi-language abstract support

---

*POC built for ContentEd Med — demonstrating AI-augmented medical publications workflow.*
