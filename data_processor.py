"""
data_processor.py
=================
PySpark Big Data processing pipeline for the Social Pulse Twitter Sentiment Analysis project.

Demonstrates genuine PySpark DataFrame transformations, group-by aggregations,
text length statistics, filtering, and sampling on the Kaggle Twitter Sentiment Dataset (160k+ records).

Architecture & Design:
1. Environment Configuration: Automatically detects JAVA_HOME (e.g. Homebrew OpenJDK 17)
   if not explicitly set in the user's shell environment.
2. SparkSession Management: Initializes a local SparkSession with tuned memory and shuffle
   partitions (master: 'local[*]' for local multi-threaded CPU processing).
3. Resilient Fallback: If PySpark fails to initialize (e.g., missing JVM or unsupported
   architecture), the module gracefully activates an explicit Pandas Fallback Engine while
   flagging the active engine status ('pyspark' vs. 'pandas_fallback') for full transparency.
4. Core Spark Transformations:
   - Schema mapping and Spark DataFrame caching.
   - Text length analysis: F.length() and character length aggregations.
   - Aggregations: groupBy('sentiment').count(), groupBy('sentiment').agg(avg, min, max).
   - Multi-criteria filtering by sentiment and keyword search.
   - Efficient sampling for table exploration and word frequency extraction without full collections.
   - Methodological comparison between dataset ground-truth labels and TextBlob predictions.
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


# Configure JAVA_HOME dynamically if needed on macOS/Homebrew
def _ensure_java_home():
    """Detects and sets JAVA_HOME if unset, checking common Homebrew OpenJDK locations."""
    if "JAVA_HOME" not in os.environ:
        candidates = [
            "/opt/homebrew/opt/openjdk@17",
            "/opt/homebrew/opt/openjdk@11",
            "/opt/homebrew/opt/openjdk",
            "/usr/local/opt/openjdk@17",
            "/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home"
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                os.environ["JAVA_HOME"] = candidate
                os.environ["PATH"] = f"{candidate}/bin:{os.environ.get('PATH', '')}"
                break

_ensure_java_home()


# Try importing PySpark
PYSPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType, DoubleType
    )
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


# Common English stopwords for word frequency analytics
COMMON_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "rt", "http", "https", "co", "amp", "get", "just",
    "like", "one", "new", "today", "will", "now", "also", "made", "make", "see",
    "people", "would", "many", "even", "time", "take", "give", "much", "know"
}


class SparkPipelineManager:
    """
    Manages SparkSession lifecycle, PySpark DataFrame operations, and resilient fallbacks.
    """

    def __init__(self):
        self.spark: Optional[Any] = None
        self.engine: str = "uninitialized"  # 'pyspark' or 'pandas_fallback'
        self.init_error: Optional[str] = None
        self._initialize_spark()

    def _initialize_spark(self):
        """Attempts to initialize a local SparkSession. Falls back to Pandas on failure."""
        if not PYSPARK_AVAILABLE:
            self.engine = "pandas_fallback"
            self.init_error = "PySpark library is not installed in the current environment."
            return

        try:
            # Set Python worker executable to match current interpreter
            os.environ["PYSPARK_PYTHON"] = sys.executable
            os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

            # Configure high-performance local SparkSession
            self.spark = (
                SparkSession.builder
                .appName("SocialPulseSentimentAnalytics")
                .master("local[*]")
                .config("spark.driver.memory", "2g")
                .config("spark.sql.shuffle.partitions", "4")
                .config("spark.ui.enabled", "false")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate()
            )
            # Reduce verbose PySpark logging to WARN
            self.spark.sparkContext.setLogLevel("WARN")
            self.engine = "pyspark"
            self.init_error = None
        except Exception as e:
            self.spark = None
            self.engine = "pandas_fallback"
            self.init_error = (
                f"SparkSession initialization failed ({type(e).__name__}: {str(e)}). "
                "Ensure Java 8/11/17 is installed and JAVA_HOME points to a valid JDK."
            )

    def is_spark_active(self) -> bool:
        """Returns True if the PySpark engine is active and operational."""
        return self.engine == "pyspark" and self.spark is not None

    def get_engine_status(self) -> Dict[str, Any]:
        """Returns metadata about the active analytics processing engine."""
        return {
            "engine": self.engine,
            "spark_version": self.spark.version if self.is_spark_active() else None,
            "master": "local[*]" if self.is_spark_active() else "In-Memory Pandas",
            "error_details": self.init_error,
            "java_home": os.environ.get("JAVA_HOME", "Not Set"),
        }

    def create_dataframe(self, pdf: pd.DataFrame) -> Any:
        """
        Converts a Pandas DataFrame into an in-memory cached PySpark DataFrame.
        If Spark is inactive, returns the Pandas DataFrame directly.
        """
        if not self.is_spark_active():
            return pdf.copy()

        # Define explicit schema for speed and stability
        schema = StructType([
            StructField("tweet_id", StringType(), False),
            StructField("tweet_text", StringType(), True),
            StructField("original_category", IntegerType(), True),
            StructField("sentiment", StringType(), True),
            StructField("char_length", IntegerType(), True),
            StructField("word_count", IntegerType(), True)
        ])

        # Select only standard fields
        cols = ["tweet_id", "tweet_text", "original_category", "sentiment", "char_length", "word_count"]
        prepared_pdf = pdf[cols].copy()
        prepared_pdf["original_category"] = prepared_pdf["original_category"].astype(int)
        prepared_pdf["char_length"] = prepared_pdf["char_length"].astype(int)
        prepared_pdf["word_count"] = prepared_pdf["word_count"].astype(int)

        spark_df = self.spark.createDataFrame(prepared_pdf, schema=schema)
        # Cache in memory for fast interactive filtering
        spark_df = spark_df.cache()
        return spark_df

    def get_kpis(self, df: Any) -> Dict[str, Any]:
        """
        Calculates high-level KPI metrics: total records, positive, negative, neutral counts,
        percentages, and average text length.
        """
        if self.is_spark_active() and hasattr(df, "groupBy"):
            total_count = df.count()
            if total_count == 0:
                return {
                    "total": 0, "positive": 0, "negative": 0, "neutral": 0,
                    "pos_pct": 0.0, "neg_pct": 0.0, "neu_pct": 0.0,
                    "avg_chars": 0.0, "avg_words": 0.0
                }

            agg_row = df.agg(
                F.count(F.when(F.col("sentiment") == "Positive", 1)).alias("positive"),
                F.count(F.when(F.col("sentiment") == "Negative", 1)).alias("negative"),
                F.count(F.when(F.col("sentiment") == "Neutral", 1)).alias("neutral"),
                F.avg("char_length").alias("avg_chars"),
                F.avg("word_count").alias("avg_words")
            ).collect()[0]

            pos = agg_row["positive"] or 0
            neg = agg_row["negative"] or 0
            neu = agg_row["neutral"] or 0
            avg_c = round(float(agg_row["avg_chars"] or 0.0), 1)
            avg_w = round(float(agg_row["avg_words"] or 0.0), 1)

            return {
                "total": total_count,
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "pos_pct": round((pos / total_count) * 100, 1),
                "neg_pct": round((neg / total_count) * 100, 1),
                "neu_pct": round((neu / total_count) * 100, 1),
                "avg_chars": avg_c,
                "avg_words": avg_w
            }
        else:
            total_count = len(df)
            if total_count == 0:
                return {
                    "total": 0, "positive": 0, "negative": 0, "neutral": 0,
                    "pos_pct": 0.0, "neg_pct": 0.0, "neu_pct": 0.0,
                    "avg_chars": 0.0, "avg_words": 0.0
                }
            pos = int((df["sentiment"] == "Positive").sum())
            neg = int((df["sentiment"] == "Negative").sum())
            neu = int((df["sentiment"] == "Neutral").sum())
            avg_c = round(float(df["char_length"].mean()), 1) if "char_length" in df.columns else 0.0
            avg_w = round(float(df["word_count"].mean()), 1) if "word_count" in df.columns else 0.0

            return {
                "total": total_count,
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "pos_pct": round((pos / total_count) * 100, 1),
                "neg_pct": round((neg / total_count) * 100, 1),
                "neu_pct": round((neu / total_count) * 100, 1),
                "avg_chars": avg_c,
                "avg_words": avg_w
            }

    def filter_data(
        self,
        df: Any,
        sentiments: Optional[List[str]] = None,
        search_query: Optional[str] = None
    ) -> Any:
        """
        Applies multi-criteria filtering on a Spark DataFrame or fallback Pandas DataFrame.
        """
        if self.is_spark_active() and hasattr(df, "filter"):
            filtered_df = df
            if sentiments:
                filtered_df = filtered_df.filter(F.col("sentiment").isin(sentiments))
            if search_query and search_query.strip():
                query = search_query.strip().lower()
                filtered_df = filtered_df.filter(F.lower(F.col("tweet_text")).contains(query))
            return filtered_df
        else:
            pdf = df.copy()
            if sentiments and "sentiment" in pdf.columns:
                pdf = pdf[pdf["sentiment"].isin(sentiments)]
            if search_query and search_query.strip() and "tweet_text" in pdf.columns:
                query = search_query.strip().lower()
                pdf = pdf[pdf["tweet_text"].str.lower().str.contains(query, na=False)]
            return pdf

    def get_sentiment_distribution(self, df: Any) -> pd.DataFrame:
        """
        Computes sentiment-wise counts and percentages.
        PySpark: df.groupBy('sentiment').count()
        """
        if self.is_spark_active() and hasattr(df, "groupBy"):
            spark_counts = (
                df.groupBy("sentiment")
                .count()
                .orderBy(F.desc("count"))
                .toPandas()
            )
            total = spark_counts["count"].sum()
            spark_counts["percentage"] = (spark_counts["count"] / total * 100).round(1) if total > 0 else 0.0
            return spark_counts
        else:
            vc = df["sentiment"].value_counts().reset_index()
            vc.columns = ["sentiment", "count"]
            total = vc["count"].sum()
            vc["percentage"] = (vc["count"] / total * 100).round(1) if total > 0 else 0.0
            return vc

    def get_length_summary(self, df: Any) -> pd.DataFrame:
        """
        Computes character and word length statistics per sentiment class.
        PySpark: df.groupBy('sentiment').agg(avg, min, max)
        """
        if self.is_spark_active() and hasattr(df, "groupBy"):
            summary = (
                df.groupBy("sentiment")
                .agg(
                    F.count("*").alias("count"),
                    F.round(F.avg("char_length"), 1).alias("avg_chars"),
                    F.min("char_length").alias("min_chars"),
                    F.max("char_length").alias("max_chars"),
                    F.round(F.avg("word_count"), 1).alias("avg_words"),
                )
                .orderBy(F.desc("count"))
                .toPandas()
            )
            return summary
        else:
            grouped = df.groupby("sentiment").agg(
                count=("tweet_id", "count"),
                avg_chars=("char_length", "mean"),
                min_chars=("char_length", "min"),
                max_chars=("char_length", "max"),
                avg_words=("word_count", "mean")
            ).reset_index()
            grouped["avg_chars"] = grouped["avg_chars"].round(1)
            grouped["avg_words"] = grouped["avg_words"].round(1)
            return grouped.sort_values("count", ascending=False)

    def get_sample_records(self, df: Any, limit: int = 100) -> pd.DataFrame:
        """
        Retrieves a bounded sample of records as a Pandas DataFrame for table exploration.
        Avoids converting the entire 160k+ DataFrame to Pandas.
        """
        if self.is_spark_active() and hasattr(df, "limit"):
            return df.limit(limit).toPandas()
        else:
            return df.head(limit)

    def get_top_frequent_words(
        self,
        df: Any,
        top_n: int = 15,
        sentiment_filter: Optional[str] = None,
        sample_size: int = 5000
    ) -> pd.DataFrame:
        """
        Extracts high-frequency words from a sample of records, excluding English stopwords.
        """
        # Sample records to keep extraction responsive
        if self.is_spark_active() and hasattr(df, "filter"):
            filtered = df
            if sentiment_filter and sentiment_filter != "All":
                filtered = filtered.filter(F.col("sentiment") == sentiment_filter)
            pdf_sample = filtered.limit(sample_size).select("tweet_text").toPandas()
        else:
            pdf_sample = df
            if sentiment_filter and sentiment_filter != "All" and "sentiment" in pdf_sample.columns:
                pdf_sample = pdf_sample[pdf_sample["sentiment"] == sentiment_filter]
            pdf_sample = pdf_sample.head(sample_size)

        if pdf_sample.empty or "tweet_text" not in pdf_sample.columns:
            return pd.DataFrame(columns=["word", "frequency"])

        word_counts: Dict[str, int] = {}
        for text in pdf_sample["tweet_text"].dropna():
            words = re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower())
            for w in words:
                if w not in COMMON_STOPWORDS:
                    word_counts[w] = word_counts.get(w, 0) + 1

        freq_df = (
            pd.DataFrame(list(word_counts.items()), columns=["word", "frequency"])
            .sort_values("frequency", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return freq_df

    def evaluate_textblob_comparison(
        self,
        df: Any,
        sample_size: int = 500,
        pos_threshold: float = 0.1,
        neg_threshold: float = -0.1
    ) -> Dict[str, Any]:
        """
        Evaluates TextBlob lexical predictions against ground-truth dataset labels on a sample.

        Returns agreement statistics and a confusion summary.
        """
        from sentiment_analyzer import batch_analyze_dataframe

        if self.is_spark_active() and hasattr(df, "limit"):
            sample_pdf = df.limit(sample_size).toPandas()
        else:
            sample_pdf = df.head(sample_size).copy()

        if sample_pdf.empty:
            return {"sample_size": 0, "agreement_rate": 0.0, "details": None}

        scored = batch_analyze_dataframe(
            sample_pdf,
            text_column="tweet_text",
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold
        )

        matches = (scored["sentiment"] == scored["predicted_sentiment"]).sum()
        agreement_rate = round(float(matches / len(scored) * 100), 1)

        crosstab = pd.crosstab(
            scored["sentiment"],
            scored["predicted_sentiment"],
            rownames=["Ground Truth"],
            colnames=["TextBlob Predicted"]
        )

        return {
            "sample_size": len(scored),
            "matches": int(matches),
            "agreement_rate": agreement_rate,
            "crosstab": crosstab,
            "sample_records": scored[["tweet_id", "tweet_text", "sentiment", "predicted_sentiment", "polarity"]].head(10)
        }
