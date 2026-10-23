"""
tests/test_sentiment.py
========================
Automated test suite for the Social Pulse Twitter Sentiment Analysis project.

Covers:
1. Sentiment Analyzer Unit Tests:
   - Clearly positive, negative, and neutral texts
   - Empty, None, and whitespace-only inputs
   - Invalid and numeric inputs
   - Text cleaning and regex normalization
   - Configurable polarity thresholds and mapping explanations
2. Dataset Loader Unit Tests:
   - Category-to-sentiment mapping (-1, 0, 1)
   - DataFrame preprocessing (null removal, deduplication, length calculations)
   - Cache file loading and schema verification
3. PySpark Big Data Processing Pipeline Tests:
   - SparkSession initialization and engine status
   - DataFrame caching and schema validation
   - High-performance KPI calculations
   - GroupBy sentiment aggregations
   - Text length summaries
   - Multi-criteria filtering and keyword search
   - Frequent word extraction and stopword exclusion
   - TextBlob vs ground-truth comparison logic
4. Individual Tweet Analysis Isolation Test:
   - Verifies single text analysis does not mutate dataset statistics.
"""

import os
import sys
import pytest
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentiment_analyzer import (
    clean_tweet,
    analyze_sentiment,
    batch_analyze_dataframe,
    DEFAULT_POS_THRESHOLD,
    DEFAULT_NEG_THRESHOLD,
)
from dataset_loader import (
    map_category_to_sentiment,
    preprocess_dataframe,
    load_dataset,
    LOCAL_CACHE_FILE
)
from data_processor import SparkPipelineManager


# ==============================================================================
# 1. SENTIMENT ANALYZER UNIT TESTS
# ==============================================================================

def test_clearly_positive_sentiment():
    """Verify that unambiguously positive sentences yield Positive classification."""
    positive_samples = [
        "This is an absolutely fantastic and delightful product! Loved every bit of it.",
        "Outstanding performance, brilliant customer support, and wonderful experience!",
        "Stunning camera quality and great battery life. Highly recommended!"
    ]
    for text in positive_samples:
        result = analyze_sentiment(text)
        assert result["sentiment"] == "Positive", f"Expected Positive for '{text}', got {result['sentiment']}"
        assert result["polarity"] > DEFAULT_POS_THRESHOLD
        assert "explanation" in result
        assert "model_note" in result


def test_clearly_negative_sentiment():
    """Verify that unambiguously negative sentences yield Negative classification."""
    negative_samples = [
        "This was a terrible, horrible, and dreadful disaster! Completely ruined my day.",
        "Worst customer service ever! Extremely rude staff and unacceptable delay.",
        "Sluggish, bloated, full of bugs, and constantly crashing. Horrible experience."
    ]
    for text in negative_samples:
        result = analyze_sentiment(text)
        assert result["sentiment"] == "Negative", f"Expected Negative for '{text}', got {result['sentiment']}"
        assert result["polarity"] < DEFAULT_NEG_THRESHOLD
        assert "explanation" in result


def test_neutral_sentiment():
    """Verify that factual and neutral statements fall within the neutral polarity range."""
    neutral_samples = [
        "The conference keynote slides have been uploaded to the public portal for review.",
        "The scheduled train departed the central railway terminal at 8:15 AM today.",
        "Specifications state a 5000 mAh battery capacity and support for USB standard."
    ]
    for text in neutral_samples:
        result = analyze_sentiment(text)
        assert result["sentiment"] == "Neutral", f"Expected Neutral for '{text}', got {result['sentiment']}"
        assert DEFAULT_NEG_THRESHOLD <= result["polarity"] <= DEFAULT_POS_THRESHOLD


def test_empty_and_whitespace_text():
    """Verify that empty, None, and whitespace-only inputs are handled safely."""
    edge_cases = ["", "   ", "\t\n", None, float("nan")]
    for val in edge_cases:
        res = analyze_sentiment(val)
        assert res["sentiment"] == "Neutral"
        assert res["polarity"] == 0.0
        assert res["cleaned_text"] == ""
        assert "explanation" in res


def test_invalid_and_numeric_input():
    """Verify non-string inputs do not crash the pipeline."""
    invalid_cases = [12345, 99.99, True, [], {}]
    for val in invalid_cases:
        res = analyze_sentiment(val)
        assert res["sentiment"] in ["Positive", "Negative", "Neutral"]
        assert isinstance(res["polarity"], float)


def test_text_cleaning_rules():
    """Verify regex cleaning strips URLs, mentions, RT headers, and preserves hashtag content."""
    raw_text = "RT @elonmusk: Breakthrough in #AI models! Check https://example.com/demo #tech"
    cleaned = clean_tweet(raw_text)
    assert "@elonmusk" not in cleaned
    assert "https://" not in cleaned
    assert "RT" not in cleaned
    assert "AI" in cleaned
    assert "tech" in cleaned
    assert "#" not in cleaned


def test_configurable_polarity_thresholds():
    """Verify that threshold adjustments modify the classification boundaries."""
    sample_text = "This is a good improvement."
    base_res = analyze_sentiment(sample_text)
    base_pol = base_res["polarity"]

    # Raising positive threshold above polarity makes it Neutral
    strict_res = analyze_sentiment(sample_text, pos_threshold=base_pol + 0.1)
    assert strict_res["sentiment"] == "Neutral"

    # Lowering positive threshold below polarity makes it Positive
    lenient_res = analyze_sentiment(sample_text, pos_threshold=base_pol - 0.1)
    assert lenient_res["sentiment"] == "Positive"


def test_batch_analyze_dataframe():
    """Verify dataframe batch processing correctly augments all required columns."""
    sample_df = pd.DataFrame({
        "tweet_id": ["TW_1", "TW_2", "TW_3"],
        "tweet_text": [
            "Great innovation and excellent design!",
            "Terrible service and awful quality.",
            "Meeting scheduled for 2 PM."
        ]
    })
    scored_df = batch_analyze_dataframe(sample_df, text_column="tweet_text")
    assert "cleaned_text" in scored_df.columns
    assert "polarity" in scored_df.columns
    assert "subjectivity" in scored_df.columns
    assert "predicted_sentiment" in scored_df.columns
    assert scored_df.iloc[0]["predicted_sentiment"] == "Positive"
    assert scored_df.iloc[1]["predicted_sentiment"] == "Negative"
    assert scored_df.iloc[2]["predicted_sentiment"] == "Neutral"


# ==============================================================================
# 2. DATASET LOADER UNIT TESTS
# ==============================================================================

def test_map_category_to_sentiment():
    """Verify correct mapping of -1, 0, 1 category values to sentiment labels."""
    assert map_category_to_sentiment(-1) == "Negative"
    assert map_category_to_sentiment(-1.0) == "Negative"
    assert map_category_to_sentiment("-1") == "Negative"

    assert map_category_to_sentiment(0) == "Neutral"
    assert map_category_to_sentiment(0.0) == "Neutral"
    assert map_category_to_sentiment("0") == "Neutral"

    assert map_category_to_sentiment(1) == "Positive"
    assert map_category_to_sentiment(1.0) == "Positive"
    assert map_category_to_sentiment("1") == "Positive"

    assert map_category_to_sentiment(99) is None
    assert map_category_to_sentiment("invalid") is None
    assert map_category_to_sentiment(None) is None


def test_preprocess_dataframe():
    """Verify preprocessing correctly drops nulls, deduplicates, and formats columns."""
    raw_sample = pd.DataFrame({
        "clean_text": [
            "Good product",
            "Bad quality",
            None,
            "   ",
            "Duplicate tweet",
            "Duplicate tweet",
            "Neutral statement"
        ],
        "category": [1, -1, 0, 1, 1, 1, 999]
    })
    cleaned_df, stats = preprocess_dataframe(raw_sample)

    assert stats["raw_count"] == 7
    assert stats["null_count"] == 1
    assert stats["empty_text_count"] == 1
    assert stats["invalid_category_count"] == 1
    assert stats["duplicate_count"] == 1
    assert len(cleaned_df) == 3

    assert set(cleaned_df.columns) == {
        "tweet_id", "tweet_text", "original_category", "sentiment", "char_length", "word_count"
    }


def test_load_dataset_from_cache():
    """Verify loading from parquet cache returns valid records with correct schema."""
    if os.path.exists(LOCAL_CACHE_FILE):
        df, meta = load_dataset()
        assert len(df) > 100000
        assert "tweet_text" in df.columns
        assert "sentiment" in df.columns
        assert set(df["sentiment"].unique()) == {"Positive", "Neutral", "Negative"}


# ==============================================================================
# 3. PYSPARK DATA PROCESSING PIPELINE TESTS
# ==============================================================================

@pytest.fixture(scope="module")
def spark_manager():
    """Fixture to share a Spark pipeline manager instance."""
    return SparkPipelineManager()


def test_pipeline_engine_status(spark_manager):
    """Verify the data processing pipeline identifies its active engine."""
    status = spark_manager.get_engine_status()
    assert status["engine"] in ["pyspark", "pandas_fallback"]


def test_spark_aggregations_and_kpis(spark_manager):
    """Verify DataFrame aggregations calculate accurate counts, distributions, and lengths."""
    test_df = pd.DataFrame({
        "tweet_id": ["T1", "T2", "T3", "T4", "T5"],
        "tweet_text": ["Great phone!", "Awful bug!", "Normal day.", "Super app!", "Boring."],
        "original_category": [1, -1, 0, 1, -1],
        "sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative"],
        "char_length": [12, 10, 11, 10, 7],
        "word_count": [2, 2, 2, 2, 1]
    })

    processed_df = spark_manager.create_dataframe(test_df)

    kpis = spark_manager.get_kpis(processed_df)
    assert kpis["total"] == 5
    assert kpis["positive"] == 2
    assert kpis["negative"] == 2
    assert kpis["neutral"] == 1
    assert kpis["pos_pct"] == 40.0
    assert kpis["neg_pct"] == 40.0
    assert kpis["neu_pct"] == 20.0

    dist_df = spark_manager.get_sentiment_distribution(processed_df)
    assert len(dist_df) == 3
    assert set(dist_df["sentiment"]) == {"Positive", "Negative", "Neutral"}

    len_df = spark_manager.get_length_summary(processed_df)
    assert len(len_df) == 3


def test_spark_filtering(spark_manager):
    """Verify multi-criteria filtering by sentiment and keyword search."""
    test_df = pd.DataFrame({
        "tweet_id": ["T1", "T2", "T3"],
        "tweet_text": ["Loving this battery life", "Hate this terrible screen", "Standard display update"],
        "original_category": [1, -1, 0],
        "sentiment": ["Positive", "Negative", "Neutral"],
        "char_length": [24, 25, 23],
        "word_count": [4, 4, 3]
    })
    processed_df = spark_manager.create_dataframe(test_df)

    filt_sent = spark_manager.filter_data(processed_df, sentiments=["Positive"])
    assert spark_manager.get_kpis(filt_sent)["total"] == 1

    filt_search = spark_manager.filter_data(processed_df, search_query="screen")
    assert spark_manager.get_kpis(filt_search)["total"] == 1


def test_frequent_words_extraction(spark_manager):
    """Verify stopword filtering and frequent word extraction."""
    test_df = pd.DataFrame({
        "tweet_id": ["T1", "T2"],
        "tweet_text": ["Awesome camera phone!", "Camera battery was awesome."],
        "original_category": [1, 1],
        "sentiment": ["Positive", "Positive"],
        "char_length": [21, 27],
        "word_count": [3, 4]
    })
    processed = spark_manager.create_dataframe(test_df)
    words_df = spark_manager.get_top_frequent_words(processed, top_n=5)
    extracted = list(words_df["word"])
    assert "camera" in extracted
    assert "awesome" in extracted
    assert "was" not in extracted


def test_textblob_comparison_evaluation(spark_manager):
    """Verify evaluation logic comparing TextBlob with ground-truth labels."""
    test_df = pd.DataFrame({
        "tweet_id": ["T1", "T2", "T3"],
        "tweet_text": ["This is wonderful and fantastic!", "Terrible and horrible service.", "Station announcement."],
        "original_category": [1, -1, 0],
        "sentiment": ["Positive", "Negative", "Neutral"],
        "char_length": [32, 30, 21],
        "word_count": [5, 4, 2]
    })
    processed = spark_manager.create_dataframe(test_df)
    comp_result = spark_manager.evaluate_textblob_comparison(processed, sample_size=3)
    assert comp_result["sample_size"] == 3
    assert comp_result["agreement_rate"] >= 0.0
    assert "crosstab" in comp_result


# ==============================================================================
# 4. INDIVIDUAL TWEET ANALYSIS ISOLATION TEST
# ==============================================================================

def test_individual_tweet_analysis_isolation(spark_manager):
    """
    Verify that analyzing an individual tweet text executes independently and does not
    mutate or alter the aggregate dataset metrics.
    """
    test_df = pd.DataFrame({
        "tweet_id": ["T1", "T2"],
        "tweet_text": ["Good morning", "Bad evening"],
        "original_category": [1, -1],
        "sentiment": ["Positive", "Negative"],
        "char_length": [12, 11],
        "word_count": [2, 2]
    })
    spark_df = spark_manager.create_dataframe(test_df)
    initial_kpis = spark_manager.get_kpis(spark_df)

    # Perform individual tweet analysis
    single_res = analyze_sentiment("This is a completely separate custom tweet entry!")
    assert single_res["sentiment"] == "Neutral" or single_res["sentiment"] in ["Positive", "Negative"]

    # Verify dataset metrics remain identical
    post_kpis = spark_manager.get_kpis(spark_df)
    assert initial_kpis["total"] == post_kpis["total"]
    assert initial_kpis["positive"] == post_kpis["positive"]
    assert initial_kpis["negative"] == post_kpis["negative"]
