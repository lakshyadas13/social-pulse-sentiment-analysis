"""
app.py
======
Social Pulse: Twitter Sentiment Analytics
Big Data Analytics Application powered by Apache PySpark, TextBlob, and Streamlit.

Design:
- Twitter/X-inspired clean, restrained editorial interface.
- White and near-white canvas (#ffffff, #f7f9f9) with charcoal typography (#0f1419, #536471).
- Twitter blue accents (#1d9bf0), subtle gray borders (#eff3f4, #cfd9de).
- Strictly zero emojis across all UI elements, labels, and metrics.
- Genuine PySpark DataFrame transformations on 160k+ real Kaggle records.
- Dedicated 'Analyze a Tweet' section for single-text sentiment classification.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local module imports
from dataset_loader import load_dataset, LOCAL_CACHE_FILE
from data_processor import SparkPipelineManager, COMMON_STOPWORDS
from sentiment_analyzer import (
    analyze_sentiment,
    DEFAULT_POS_THRESHOLD,
    DEFAULT_NEG_THRESHOLD,
)


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Social Pulse | Twitter Sentiment Analytics",
    page_icon="https://abs.twimg.com/favicons/twitter.3.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Twitter/X-inspired editorial design CSS
TWITTER_THEME_CSS = """
<style>
    /* Global background and typography */
    .stApp {
        background-color: #ffffff;
        color: #0f1419;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #0f1419 !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em;
    }

    /* Header section */
    .header-container {
        padding: 16px 0 20px 0;
        border-bottom: 1px solid #eff3f4;
        margin-bottom: 24px;
    }

    .header-title {
        font-size: 26px;
        font-weight: 800;
        color: #0f1419;
        margin: 0;
        line-height: 1.2;
    }

    .header-subtitle {
        font-size: 14px;
        color: #536471;
        margin-top: 4px;
        font-weight: 400;
    }

    .header-meta-row {
        display: flex;
        gap: 12px;
        align-items: center;
        margin-top: 10px;
        flex-wrap: wrap;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .badge-source {
        background-color: #f7f9f9;
        color: #536471;
        border: 1px solid #cfd9de;
    }

    .badge-spark {
        background-color: #e8f5fd;
        color: #1d9bf0;
        border: 1px solid #c7e5f9;
    }

    .badge-fallback {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    /* Metrics Cards */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #eff3f4;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #536471;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #0f1419;
        line-height: 1.1;
    }

    .metric-subtext {
        font-size: 12px;
        color: #536471;
        margin-top: 4px;
    }

    /* Analyze Tweet Box */
    .tweet-analyzer-card {
        background-color: #f7f9f9;
        border: 1px solid #cfd9de;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 24px;
    }

    .result-panel {
        background-color: #ffffff;
        border: 1px solid #eff3f4;
        border-radius: 10px;
        padding: 16px;
        margin-top: 16px;
    }

    .sentiment-tag-positive {
        display: inline-block;
        background-color: #e6f9f0;
        color: #00ba7c;
        border: 1px solid #a3e9c7;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
    }

    .sentiment-tag-negative {
        display: inline-block;
        background-color: #feecee;
        color: #f4212e;
        border: 1px solid #fca5a5;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
    }

    .sentiment-tag-neutral {
        display: inline-block;
        background-color: #f3f4f6;
        color: #4b5563;
        border: 1px solid #d1d5db;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
    }

    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eff3f4;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #eff3f4;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 4px;
        font-weight: 600;
        color: #536471;
    }

    .stTabs [aria-selected="true"] {
        color: #0f1419 !important;
        border-bottom: 3px solid #1d9bf0 !important;
    }

    /* Streamlit buttons */
    .stButton>button {
        background-color: #0f1419;
        color: #ffffff;
        border: none;
        border-radius: 9999px;
        font-weight: 700;
        padding: 9px 24px;
        font-size: 14px;
        white-space: nowrap !important;
        min-width: fit-content !important;
        width: auto !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: background-color 0.15s ease;
    }

    .stButton>button p {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        margin: 0 !important;
    }

    .stButton>button:hover {
        background-color: #272c30;
        color: #ffffff;
    }

    .stDownloadButton>button {
        background-color: #ffffff;
        color: #0f1419;
        border: 1px solid #cfd9de;
        border-radius: 9999px;
        font-weight: 700;
        padding: 8px 22px;
        font-size: 14px;
        white-space: nowrap !important;
        min-width: fit-content !important;
        width: auto !important;
    }
    .stDownloadButton>button p {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    .stDownloadButton>button:hover {
        background-color: #f7f9f9;
        border-color: #536471;
    }
</style>
"""
st.markdown(TWITTER_THEME_CSS, unsafe_allow_html=True)


# ==============================================================================
# CACHED RESOURCE INITIALIZATION
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_pipeline_manager() -> SparkPipelineManager:
    """Initializes and caches the local PySpark SparkSession manager."""
    return SparkPipelineManager()


@st.cache_data(show_spinner=False)
def get_dataset():
    """Loads the preprocessed Kaggle Twitter Sentiment Dataset from Parquet cache."""
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_cached_spark_dataframe(_manager: SparkPipelineManager, _df_full: pd.DataFrame):
    """
    Constructs and caches the PySpark DataFrame once across application lifecycle.
    Prevents expensive conversions of 160k+ rows on every user interaction.
    """
    return _manager.create_dataframe(_df_full)


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
def main():
    manager = get_pipeline_manager()
    engine_status = manager.get_engine_status()

    # Load Kaggle Dataset (reads instantly from memory or local parquet cache)
    try:
        df_full, dataset_meta = get_dataset()
    except Exception as e:
        st.error(
            f"Dataset Loading Error: {str(e)}\n\n"
            "Troubleshooting:\n"
            "1. Verify internet connectivity to download the Kaggle dataset.\n"
            "2. Ensure 'kagglehub' is installed: pip install kagglehub\n"
            "3. If using a private dataset, configure your Kaggle API key (kaggle.json)."
        )
        return

    # Ingest into PySpark DataFrame (cached once across reruns)
    spark_df_full = get_cached_spark_dataframe(manager, df_full)

    # --------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------------------------------
    st.sidebar.markdown("### Controls and Filters")
    st.sidebar.markdown("---")

    # Sentiment Filter
    sentiment_options = ["Positive", "Neutral", "Negative"]
    selected_sentiments = st.sidebar.multiselect(
        "Filter by Sentiment:",
        options=sentiment_options,
        default=sentiment_options,
        help="Select one or more ground-truth sentiment classes to filter aggregate views."
    )

    # Search keyword
    search_keyword = st.sidebar.text_input(
        "Search Dataset:",
        value="",
        placeholder="Search keywords...",
        help="Filters the dataset by matching words in tweet text."
    )

    # Configurable Thresholds for Individual Tweet Analysis
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Sensitivity")
    st.sidebar.caption("Configurable boundaries for TextBlob polarity mapping:")
    pos_thresh = st.sidebar.slider(
        "Positive Threshold (>)",
        min_value=0.01,
        max_value=0.50,
        value=DEFAULT_POS_THRESHOLD,
        step=0.01,
        help="Tweets with polarity strictly greater than this value are classified as Positive."
    )
    neg_thresh = st.sidebar.slider(
        "Negative Threshold (<)",
        min_value=-0.50,
        max_value=-0.01,
        value=DEFAULT_NEG_THRESHOLD,
        step=0.01,
        help="Tweets with polarity strictly less than this value are classified as Negative."
    )

    # Cache Control
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Cache")
    st.sidebar.caption(f"Source: {dataset_meta.get('source')}")
    st.sidebar.caption(f"Cached File: {LOCAL_CACHE_FILE}")
    if st.sidebar.button("Reload Dataset"):
        st.cache_data.clear()
        st.rerun()

    # --------------------------------------------------------------------------
    # HEADER SECTION
    # --------------------------------------------------------------------------
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">Social Pulse</h1>
        <div class="header-subtitle">Twitter Sentiment Analytics</div>
        <div class="header-meta-row">
            <span class="badge-pill badge-source">Source: Kaggle (saurabhshahane/twitter-sentiment-dataset)</span>
            <span class="badge-pill {'badge-spark' if manager.is_spark_active() else 'badge-fallback'}">
                {'Engine: Apache PySpark ' + str(engine_status.get('spark_version', '')) + ' (local[*])' if manager.is_spark_active() else 'Engine: In-Memory Pandas Fallback'}
            </span>
            <span class="badge-pill badge-source">{len(df_full):,} Valid Records</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # INDIVIDUAL TWEET ANALYSIS SECTION ("Analyze a Tweet")
    # --------------------------------------------------------------------------
    st.markdown("### Analyze a Tweet")
    st.caption("Perform on-demand sentiment classification on an individual tweet text without altering dataset aggregates.")

    with st.container():
        input_tweet_text = st.text_area(
            "Enter tweet text:",
            height=90,
            placeholder="Type or paste any tweet text here to evaluate its sentiment...",
            label_visibility="collapsed"
        )

        analyze_clicked = st.button("Analyze Sentiment", key="btn_analyze_tweet", use_container_width=False)

        if analyze_clicked:
            if not input_tweet_text or not input_tweet_text.strip():
                st.warning("Please enter valid tweet text before running analysis.")
            else:
                result = analyze_sentiment(
                    input_tweet_text,
                    pos_threshold=pos_thresh,
                    neg_threshold=neg_thresh
                )

                sent_tag_class = (
                    "sentiment-tag-positive" if result["sentiment"] == "Positive"
                    else "sentiment-tag-negative" if result["sentiment"] == "Negative"
                    else "sentiment-tag-neutral"
                )

                st.markdown(f"""
                <div class="result-panel">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span class="{sent_tag_class}">{result['sentiment']}</span>
                        <span style="font-size: 13px; color: #536471; font-weight: 600;">
                            Polarity Score: <strong>{result['polarity']:+.4f}</strong> &nbsp;|&nbsp; Subjectivity: <strong>{result['subjectivity']:.4f}</strong>
                        </span>
                    </div>
                    <div style="font-size: 14px; color: #0f1419; margin-bottom: 10px; padding: 10px; background-color: #f7f9f9; border-radius: 8px;">
                        <em>"{result['cleaned_text']}"</em>
                    </div>
                    <div style="font-size: 13px; color: #536471; line-height: 1.4;">
                        <strong>Mapping Explanation:</strong> {result['explanation']}
                    </div>
                    <div style="font-size: 12px; color: #8899a6; margin-top: 8px;">
                        <strong>Note:</strong> {result['model_note']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # BIG DATASET ANALYTICS SECTION
    # --------------------------------------------------------------------------
    # Apply PySpark Filtering
    filtered_df = manager.filter_data(
        spark_df_full,
        sentiments=selected_sentiments if selected_sentiments else None,
        search_query=search_keyword if search_keyword else None
    )

    kpis = manager.get_kpis(filtered_df)

    if kpis["total"] == 0:
        st.warning("No records match the current sidebar filter settings.")
        return

    # Dataset Overview Metrics
    st.markdown("### Dataset Overview")
    st.caption(f"Real analytics calculated from {kpis['total']:,} filtered records using Apache PySpark.")

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Tweets</div>
            <div class="metric-value">{kpis['total']:,}</div>
            <div class="metric-subtext">Cleaned & Validated</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Positive Tweets</div>
            <div class="metric-value" style="color: #00ba7c;">{kpis['positive']:,}</div>
            <div class="metric-subtext">{kpis['pos_pct']}% of dataset</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Neutral Tweets</div>
            <div class="metric-value" style="color: #536471;">{kpis['neutral']:,}</div>
            <div class="metric-subtext">{kpis['neu_pct']}% of dataset</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Negative Tweets</div>
            <div class="metric-value" style="color: #f4212e;">{kpis['negative']:,}</div>
            <div class="metric-subtext">{kpis['neg_pct']}% of dataset</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Average Text Length</div>
            <div class="metric-value">{kpis['avg_chars']} chars</div>
            <div class="metric-subtext">{kpis['avg_words']} words/tweet</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Visualizations Tabbed View
    tab_analytics, tab_explorer, tab_comparison, tab_upload = st.tabs([
        "Sentiment Distribution & Analysis",
        "Tweet Explorer",
        "TextBlob vs Dataset Comparison",
        "Upload and Analyze Dataset"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: SENTIMENT DISTRIBUTION & DETAILED ANALYSIS
    # --------------------------------------------------------------------------
    with tab_analytics:
        col_c1, col_c2 = st.columns([1, 1.2])

        sentiment_palette = {
            "Positive": "#00ba7c",  # Twitter Green
            "Neutral": "#536471",   # Twitter Slate / Charcoal
            "Negative": "#f4212e"   # Twitter Red
        }

        with col_c1:
            st.markdown("#### Sentiment Distribution")
            dist_df = manager.get_sentiment_distribution(filtered_df)

            fig_donut = go.Figure(data=[go.Pie(
                labels=dist_df["sentiment"],
                values=dist_df["count"],
                hole=0.6,
                marker=dict(colors=[sentiment_palette.get(s, "#cfd9de") for s in dist_df["sentiment"]]),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                insidetextorientation="horizontal"
            )])
            fig_donut.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#0f1419", size=13),
                margin=dict(l=10, r=10, t=20, b=20),
                height=320,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_c2:
            st.markdown("#### Tweet Volume by Sentiment")
            fig_bar = px.bar(
                dist_df,
                x="sentiment",
                y="count",
                color="sentiment",
                color_discrete_map=sentiment_palette,
                text="count",
                labels={"sentiment": "Sentiment Category", "count": "Record Count"}
            )
            fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_bar.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#0f1419"),
                xaxis=dict(showgrid=False, linecolor="#eff3f4"),
                yaxis=dict(gridcolor="#eff3f4", linecolor="#eff3f4"),
                margin=dict(l=10, r=10, t=20, b=20),
                height=320,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # Text Length Distribution & High-Frequency Words
        col_l1, col_l2 = st.columns([1, 1])

        with col_l1:
            st.markdown("#### Text Length Analysis by Sentiment")
            st.caption("Calculated using PySpark DataFrame aggregations:")
            length_df = manager.get_length_summary(filtered_df)
            st.dataframe(
                length_df.rename(columns={
                    "sentiment": "Sentiment",
                    "count": "Total Tweets",
                    "avg_chars": "Avg Characters",
                    "min_chars": "Min Chars",
                    "max_chars": "Max Chars",
                    "avg_words": "Avg Words"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("""
            <div style="font-size: 13px; color: #536471; background-color: #f7f9f9; padding: 12px; border-radius: 8px; border: 1px solid #eff3f4; margin-top: 10px;">
                <strong>Dataset Schema Note:</strong> The Kaggle dataset contains <code>clean_text</code> and 
                <code>category</code> fields only. Domain topic classifications, geographic tags, and timestamps 
                are not present in this dataset and are not fabricated.
            </div>
            """, unsafe_allow_html=True)

        with col_l2:
            st.markdown("#### High-Frequency Keywords")
            word_filter = st.selectbox(
                "Filter words by sentiment:",
                ["All", "Positive", "Neutral", "Negative"],
                index=0
            )

            top_words_df = manager.get_top_frequent_words(
                filtered_df,
                top_n=12,
                sentiment_filter=word_filter
            )

            if not top_words_df.empty:
                bar_color = sentiment_palette.get(word_filter, "#1d9bf0")
                fig_words = px.bar(
                    top_words_df,
                    x="frequency",
                    y="word",
                    orientation="h",
                    text="frequency",
                    labels={"frequency": "Occurrences", "word": "Keyword"}
                )
                fig_words.update_traces(marker_color=bar_color, textposition="outside")
                fig_words.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#0f1419"),
                    yaxis=dict(autorange="reversed", showgrid=False, linecolor="#eff3f4"),
                    xaxis=dict(gridcolor="#eff3f4", linecolor="#eff3f4"),
                    margin=dict(l=10, r=10, t=10, b=20),
                    height=280
                )
                st.plotly_chart(fig_words, use_container_width=True)
            else:
                st.info("No frequent words extracted for the current selection.")

    # --------------------------------------------------------------------------
    # TAB 2: TWEET EXPLORER TABLE
    # --------------------------------------------------------------------------
    with tab_explorer:
        st.markdown("#### Tweet Explorer")
        st.caption("Inspect and search actual records from the Kaggle dataset. Filtered using PySpark:")

        col_sample_size, _ = st.columns([1, 3])
        with col_sample_size:
            display_limit = st.selectbox("Display Limit:", [50, 100, 250, 500], index=1)

        sample_records_df = manager.get_sample_records(filtered_df, limit=display_limit)

        st.caption(f"Showing {len(sample_records_df):,} records out of {kpis['total']:,} filtered records.")

        st.dataframe(
            sample_records_df[[
                "tweet_id",
                "tweet_text",
                "original_category",
                "sentiment",
                "char_length",
                "word_count"
            ]].rename(columns={
                "tweet_id": "Tweet ID",
                "tweet_text": "Tweet Text",
                "original_category": "Raw Label (-1/0/1)",
                "sentiment": "Ground-Truth Sentiment",
                "char_length": "Characters",
                "word_count": "Words"
            }),
            use_container_width=True,
            height=380,
            hide_index=True
        )

        csv_bytes = sample_records_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Sample as CSV",
            data=csv_bytes,
            file_name=f"social_pulse_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # --------------------------------------------------------------------------
    # TAB 3: TEXTBLOB VS DATASET GROUND TRUTH COMPARISON
    # --------------------------------------------------------------------------
    with tab_comparison:
        st.markdown("#### TextBlob vs Dataset Ground-Truth Comparison")
        st.caption(
            "Methodological evaluation: Compares TextBlob lexical predictions with human-annotated "
            "ground-truth labels on a representative sample of the dataset."
        )

        col_samp, col_btn_run = st.columns([1, 1])
        with col_samp:
            comp_sample_size = st.slider("Evaluation Sample Size:", min_value=100, max_value=1000, value=300, step=100)
        with col_btn_run:
            run_comp = st.button("Run Evaluation Comparison")

        if run_comp:
            with st.spinner("Evaluating TextBlob against dataset labels..."):
                comp_result = manager.evaluate_textblob_comparison(
                    filtered_df,
                    sample_size=comp_sample_size,
                    pos_threshold=pos_thresh,
                    neg_threshold=neg_thresh
                )

            st.markdown(f"""
            <div style="display: flex; gap: 16px; margin: 16px 0;">
                <div class="metric-card" style="flex: 1;">
                    <div class="metric-label">Evaluated Sample</div>
                    <div class="metric-value">{comp_result['sample_size']}</div>
                </div>
                <div class="metric-card" style="flex: 1;">
                    <div class="metric-label">Agreement Count</div>
                    <div class="metric-value">{comp_result['matches']}</div>
                </div>
                <div class="metric-card" style="flex: 1;">
                    <div class="metric-label">Agreement Rate</div>
                    <div class="metric-value" style="color: #1d9bf0;">{comp_result['agreement_rate']}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Confusion Matrix (Ground-Truth vs Predicted)")
            st.dataframe(comp_result["crosstab"], use_container_width=True)

            st.markdown("##### Sample Classified Records")
            st.dataframe(
                comp_result["sample_records"].rename(columns={
                    "tweet_id": "Tweet ID",
                    "tweet_text": "Tweet Text",
                    "sentiment": "Ground Truth",
                    "predicted_sentiment": "TextBlob Predicted",
                    "polarity": "Polarity"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("""
            <div style="font-size: 13px; color: #536471; background-color: #f7f9f9; padding: 12px; border-radius: 8px; border: 1px solid #eff3f4; margin-top: 10px;">
                <strong>Methodological Insight:</strong> Discrepancies between ground-truth labels and TextBlob 
                arise because TextBlob uses a general English lexical dictionary. It does not account for political 
                irony, contextual colloquialisms, or domain-specific sentiment nuances present in human annotations.
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 4: UPLOAD AND ANALYZE CUSTOM DATASET
    # --------------------------------------------------------------------------
    with tab_upload:
        st.markdown("#### Upload and Analyze Custom Dataset")
        st.caption(
            "Upload any external CSV dataset (e.g. social posts, product reviews, feedback surveys) "
            "to generate sentiment predictions, polarity scores, and interactive analytics."
        )

        uploaded_file = st.file_uploader(
            "Select CSV file:",
            type=["csv"],
            help="Upload a CSV dataset containing text columns to evaluate.",
            key="custom_dataset_uploader"
        )

        if uploaded_file is not None:
            try:
                # Read sample to inspect schema
                preview_df = pd.read_csv(uploaded_file, nrows=5)
                cols = list(preview_df.columns)
                uploaded_file.seek(0)

                with st.expander("Dataset Preview (First 5 Rows)", expanded=False):
                    st.dataframe(preview_df, use_container_width=True)

                col_u1, col_u2 = st.columns([1.5, 1])
                with col_u1:
                    # Auto-detect text column
                    text_candidates = [
                        c for c in cols if c.lower() in [
                            "clean_text", "tweet_text", "text", "tweet", "content",
                            "comment", "review", "body", "message", "feedback", "sentence"
                        ]
                    ]
                    default_text_idx = cols.index(text_candidates[0]) if text_candidates else 0
                    selected_text_col = st.selectbox(
                        "Column containing text to analyze:",
                        options=cols,
                        index=default_text_idx,
                        key="custom_text_col_select"
                    )

                with col_u2:
                    max_records_opt = st.select_slider(
                        "Maximum records to analyze:",
                        options=[250, 500, 1000, 2500, 5000, "All Records"],
                        value=1000,
                        help="Limits processing size for faster execution on large files."
                    )

                if st.button("Generate Sentiment Analysis", key="btn_run_uploaded_analysis"):
                    with st.spinner("Processing text and calculating sentiment scores..."):
                        full_uploaded_df = pd.read_csv(uploaded_file)
                        
                        # Filter to requested sample if not "All Records"
                        if isinstance(max_records_opt, int) and len(full_uploaded_df) > max_records_opt:
                            full_uploaded_df = full_uploaded_df.head(max_records_opt).copy()

                        # Drop nulls in selected text column
                        clean_input_df = full_uploaded_df.dropna(subset=[selected_text_col]).copy()
                        clean_input_df[selected_text_col] = clean_input_df[selected_text_col].astype(str)

                        from sentiment_analyzer import batch_analyze_dataframe
                        scored_custom_df = batch_analyze_dataframe(
                            clean_input_df,
                            text_column=selected_text_col,
                            pos_threshold=pos_thresh,
                            neg_threshold=neg_thresh
                        )
                        st.session_state["custom_analyzed_df"] = scored_custom_df
                        st.session_state["custom_analyzed_name"] = uploaded_file.name

            except Exception as e:
                st.error(f"Error parsing uploaded file: {str(e)}")

        # Display results if analyzed data exists in session_state
        if "custom_analyzed_df" in st.session_state:
            cust_df = st.session_state["custom_analyzed_df"]
            dataset_name = st.session_state.get("custom_analyzed_name", "Uploaded Dataset")

            st.markdown("---")
            st.markdown(f"##### Results for: {dataset_name}")

            total_cust = len(cust_df)
            pos_cust = int((cust_df["predicted_sentiment"] == "Positive").sum())
            neu_cust = int((cust_df["predicted_sentiment"] == "Neutral").sum())
            neg_cust = int((cust_df["predicted_sentiment"] == "Negative").sum())
            avg_pol_cust = round(float(cust_df["polarity"].mean()), 3) if total_cust > 0 else 0.0

            pos_pct_cust = round((pos_cust / total_cust) * 100, 1) if total_cust > 0 else 0.0
            neu_pct_cust = round((neu_cust / total_cust) * 100, 1) if total_cust > 0 else 0.0
            neg_pct_cust = round((neg_cust / total_cust) * 100, 1) if total_cust > 0 else 0.0

            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Analyzed Records</div>
                    <div class="metric-value">{total_cust:,}</div>
                    <div class="metric-subtext">Total Processed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Positive Sentiment</div>
                    <div class="metric-value" style="color: #00ba7c;">{pos_cust:,}</div>
                    <div class="metric-subtext">{pos_pct_cust}% of dataset</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Neutral Sentiment</div>
                    <div class="metric-value" style="color: #536471;">{neu_cust:,}</div>
                    <div class="metric-subtext">{neu_pct_cust}% of dataset</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Negative Sentiment</div>
                    <div class="metric-value" style="color: #f4212e;">{neg_cust:,}</div>
                    <div class="metric-subtext">{neg_pct_cust}% of dataset</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Polarity</div>
                    <div class="metric-value">{avg_pol_cust:+.3f}</div>
                    <div class="metric-subtext">Scale: [-1.0 to +1.0]</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Charts for uploaded dataset
            col_chart_u1, col_chart_u2 = st.columns([1, 1.2])

            sentiment_palette = {
                "Positive": "#00ba7c",
                "Neutral": "#536471",
                "Negative": "#f4212e"
            }

            dist_cust_data = pd.DataFrame({
                "sentiment": ["Positive", "Neutral", "Negative"],
                "count": [pos_cust, neu_cust, neg_cust]
            })

            with col_chart_u1:
                st.markdown("###### Sentiment Proportions")
                fig_cust_pie = go.Figure(data=[go.Pie(
                    labels=dist_cust_data["sentiment"],
                    values=dist_cust_data["count"],
                    hole=0.6,
                    marker=dict(colors=[sentiment_palette.get(s, "#cfd9de") for s in dist_cust_data["sentiment"]]),
                    textinfo="label+percent",
                    hoverinfo="label+value+percent"
                )])
                fig_cust_pie.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#0f1419", size=13),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=280,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_cust_pie, use_container_width=True)

            with col_chart_u2:
                st.markdown("###### Sentiment Counts")
                fig_cust_bar = px.bar(
                    dist_cust_data,
                    x="sentiment",
                    y="count",
                    color="sentiment",
                    color_discrete_map=sentiment_palette,
                    text="count",
                    labels={"sentiment": "Sentiment", "count": "Record Count"}
                )
                fig_cust_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
                fig_cust_bar.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#0f1419"),
                    xaxis=dict(showgrid=False, linecolor="#eff3f4"),
                    yaxis=dict(gridcolor="#eff3f4", linecolor="#eff3f4"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=280,
                    showlegend=False
                )
                st.plotly_chart(fig_cust_bar, use_container_width=True)

            # Table exploration
            st.markdown("###### Analyzed Records Explorer")
            col_search_u, col_filt_u = st.columns([2, 1])
            with col_search_u:
                search_cust_query = st.text_input(
                    "Search in uploaded records:",
                    value="",
                    placeholder="Search keywords in text...",
                    key="custom_search_input"
                )
            with col_filt_u:
                filter_cust_sent = st.multiselect(
                    "Filter by sentiment:",
                    options=["Positive", "Neutral", "Negative"],
                    default=["Positive", "Neutral", "Negative"],
                    key="custom_sentiment_filter"
                )

            filtered_cust_table = cust_df[cust_df["predicted_sentiment"].isin(filter_cust_sent)].copy()
            if search_cust_query.strip():
                query_str = search_cust_query.strip().lower()
                filtered_cust_table = filtered_cust_table[
                    filtered_cust_table["cleaned_text"].str.lower().str.contains(query_str, na=False)
                ]

            st.caption(f"Showing {len(filtered_cust_table):,} records out of {total_cust:,} analyzed.")

            display_cols_cust = ["cleaned_text", "polarity", "subjectivity", "predicted_sentiment"]
            st.dataframe(
                filtered_cust_table[display_cols_cust].rename(columns={
                    "cleaned_text": "Cleaned Text",
                    "polarity": "Polarity",
                    "subjectivity": "Subjectivity",
                    "predicted_sentiment": "Predicted Sentiment"
                }),
                use_container_width=True,
                height=320,
                hide_index=True
            )

            # Download analyzed dataset button
            csv_cust_bytes = cust_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Analyzed Dataset as CSV",
                data=csv_cust_bytes,
                file_name=f"social_pulse_analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="btn_download_custom_analysis"
            )


if __name__ == "__main__":
    main()
