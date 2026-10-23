# Social Pulse: Twitter Sentiment Analytics

An enterprise-grade Big Data Analytics (BDA) application that ingests, cleans, classifies, and visualizes social media sentiment across 160,000+ real Twitter records using Apache PySpark, TextBlob, kagglehub, and Streamlit.

---

## Key Capabilities

* Real Data Source via kagglehub: Programmatic download, preprocessing, and local Parquet caching of the public Kaggle Twitter Sentiment Dataset (`saurabhshahane/twitter-sentiment-dataset`, 160,843 verified records).
* Apache PySpark Big Data Pipeline: Genuine PySpark DataFrame transformations (`groupBy`, `agg`, `filter`, and length statistics) running on a local multi-threaded Spark master (`local[*]`).
* Resilient Engine Fallback: Seamless transition to an in-memory Pandas engine with diagnostic reporting if Java or Spark dependencies are unavailable.
* Individual Tweet Sentiment Analysis: A dedicated "Analyze a Tweet" interface allowing users to evaluate single text inputs in real time using TextBlob lexical polarity and configurable thresholds, isolated from dataset aggregates.
* Twitter/X-Inspired Interface: Clean editorial design featuring a white/light canvas, charcoal typography, subtle gray dividers, and Twitter blue accents with zero emojis.
* Methodological Evaluation: Cross-tabulated evaluation comparing TextBlob lexical predictions against human-annotated ground-truth labels on sample populations.
* Automated Test Suite: 17 comprehensive unit and integration tests with 100% pass rate.

---

## Technology Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| Programming Language | Python | 3.10+ | Core pipeline and application logic |
| Big Data Analytics | Apache PySpark | 4.2.0 | Distributed DataFrame processing, transformations, and aggregations |
| Dataset Management | kagglehub | 1.0.3 | Automated dataset download from Kaggle |
| In-Memory Data | Pandas | 3.1.0 | Local Parquet caching, tabular manipulation, and fallback engine |
| Sentiment Engine | TextBlob | 0.20.1 | Lexical polarity and subjectivity computation |
| Web Application | Streamlit | 1.66.1 | Interactive analytical web dashboard |
| Visualizations | Plotly | 7.1.0 | Interactive donut, bar, and keyword charts |
| Testing | Pytest | 9.1.1 | Unit and pipeline integration tests |

---

## Project Structure

```text
twitter-sentiment-analysis/
│
├── app.py                     # Streamlit & Plotly Twitter/X-inspired dashboard
├── dataset_loader.py          # Kagglehub ingestion, cleaning, mapping & caching
├── sentiment_analyzer.py      # Preprocessing, TextBlob polarity & threshold mapping
├── data_processor.py          # PySpark SparkSession, DataFrame aggregations & fallback
├── requirements.txt           # Project dependencies
├── README.md                  # Concise setup & architectural overview
├── PROJECT_DOCUMENTATION.md   # Comprehensive technical report & viva handbook
│
├── data/
│   └── processed_dataset.parquet # Cleaned and cached dataset (160,843 rows)
│
└── tests/
    └── test_sentiment.py      # Automated test suite (17 tests)
```

---

## Installation & Setup

### 1. Prerequisites
* Python: Version 3.10 to 3.14.
* Java Runtime (for PySpark): OpenJDK 8, 11, or 17.
  * macOS (Homebrew): `brew install openjdk@17`
  * Ubuntu / Debian: `sudo apt install openjdk-17-jdk`
  * Windows: Install OpenJDK 17 from Eclipse Temurin and configure `JAVA_HOME`.
* Kaggle Access: The dataset `saurabhshahane/twitter-sentiment-dataset` is public and does not require credentials. For private datasets, ensure `~/.kaggle/kaggle.json` exists.

### 2. Setup Virtual Environment
```bash
# Clone or navigate to the repository
cd "/path/to/twitter"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate       # Windows PowerShell
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Execution Guide

### Step 1: Preload & Cache Dataset (Optional)
```bash
python dataset_loader.py
```
*Downloads the dataset via kagglehub, validates 160,843 records, and creates `data/processed_dataset.parquet`.*

### Step 2: Run Automated Tests
```bash
pytest tests/test_sentiment.py -v
```
*Executes all 17 unit, data pipeline, and Spark aggregation test cases.*

### Step 3: Launch the Application
```bash
streamlit run app.py
```
*Opens the dashboard at `http://localhost:8501`.*

---

## Detailed Technical Documentation

For the full technical report, architecture diagrams, mathematical sentiment formulas, Spark execution plans, and 20+ viva examination questions with complete answers, see:

[PROJECT_DOCUMENTATION.md](file:///Users/lakshyadas/Downloads/work%20work%20work/Projects/twitter/PROJECT_DOCUMENTATION.md)
