"""
dataset_loader.py
=================
Kaggle dataset loader and preprocessor for the Social Pulse Twitter Sentiment Analysis project.

Dataset: saurabhshahane/twitter-sentiment-dataset (via kagglehub)
Actual schema:
  - clean_text: string (raw/cleaned tweet text)
  - category: float/int (-1.0 = Negative, 0.0 = Neutral, 1.0 = Positive)

Functionality:
  1. Programmatically downloads the dataset using kagglehub.
  2. Dynamically locates the CSV file inside the downloaded directory.
  3. Preprocesses and validates records: drops nulls, deduplicates, and strips whitespace.
  4. Normalizes original category codes into human-readable sentiment labels.
  5. Caches the cleaned dataset locally in Parquet format for fast subsequent loads.
"""

import os
import glob
from typing import Optional, Tuple, Dict, Any
import pandas as pd

KAGGLE_DATASET_ID = "saurabhshahane/twitter-sentiment-dataset"
LOCAL_CACHE_DIR = "data"
LOCAL_CACHE_FILE = os.path.join(LOCAL_CACHE_DIR, "processed_dataset.parquet")


def download_kaggle_dataset(dataset_id: str = KAGGLE_DATASET_ID) -> str:
    """
    Downloads the specified dataset from Kaggle via kagglehub and locates the CSV file.

    Parameters:
        dataset_id: Kaggle dataset identifier (e.g., 'saurabhshahane/twitter-sentiment-dataset')

    Returns:
        str: Absolute path to the identified CSV file.

    Raises:
        RuntimeError: If download fails or no CSV file is found.
    """
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            "kagglehub is not installed. Install it via 'pip install kagglehub'."
        ) from e

    try:
        download_path = kagglehub.dataset_download(dataset_id)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Kaggle dataset '{dataset_id}' via kagglehub. "
            f"Error details: {str(e)}. "
            "Ensure an active internet connection is available. If using a private dataset, "
            "verify Kaggle API credentials (kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY)."
        ) from e

    if not download_path or not os.path.exists(download_path):
        raise RuntimeError(f"Download path does not exist: {download_path}")

    # Search for CSV files in the downloaded directory tree
    csv_candidates = glob.glob(os.path.join(download_path, "**", "*.csv"), recursive=True)
    if not csv_candidates:
        raise RuntimeError(
            f"No CSV file found in downloaded Kaggle directory: {download_path}"
        )

    # Pick the largest CSV file if multiple exist
    csv_candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    selected_csv = csv_candidates[0]
    return selected_csv


def map_category_to_sentiment(category_val: Any) -> Optional[str]:
    """
    Maps original dataset category values (-1, 0, 1) to normalized sentiment strings.

    Mapping:
      -1 / -1.0 / '-1' -> 'Negative'
       0 /  0.0 / '0'  -> 'Neutral'
       1 /  1.0 / '1'  -> 'Positive'

    Returns:
        Optional[str]: 'Negative', 'Neutral', 'Positive', or None if invalid.
    """
    try:
        val = int(float(category_val))
        if val == -1:
            return "Negative"
        elif val == 0:
            return "Neutral"
        elif val == 1:
            return "Positive"
    except (ValueError, TypeError):
        pass
    return None


def preprocess_dataframe(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cleans raw DataFrame from Twitter_Data.csv and standardizes schema.

    Cleaning steps:
      1. Detect text and category columns.
      2. Record initial raw count.
      3. Drop rows with null text or null category.
      4. Filter out whitespace-only text.
      5. Map category values to integer and normalized label.
      6. Remove duplicate text entries.
      7. Generate sequential tweet identifiers.

    Parameters:
        raw_df: Raw Pandas DataFrame loaded from CSV.

    Returns:
        Tuple[pd.DataFrame, Dict[str, int]]: (Processed DataFrame, Cleaning Statistics)
    """
    initial_count = len(raw_df)

    # Identify text column
    text_candidates = ["clean_text", "tweet_text", "text", "tweet"]
    text_col = None
    for col in text_candidates:
        if col in raw_df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = raw_df.columns[0]

    # Identify category column
    cat_candidates = ["category", "sentiment", "label", "target"]
    cat_col = None
    for col in cat_candidates:
        if col in raw_df.columns:
            cat_col = col
            break
    if cat_col is None:
        cat_col = raw_df.columns[1] if len(raw_df.columns) > 1 else raw_df.columns[0]

    df = raw_df[[text_col, cat_col]].copy()
    df.columns = ["tweet_text", "category"]

    # Drop nulls
    null_count = df.isna().any(axis=1).sum()
    df = df.dropna().copy()

    # Normalize text to string and strip whitespace
    df["tweet_text"] = df["tweet_text"].astype(str).str.strip()
    empty_text_mask = df["tweet_text"] == ""
    empty_count = empty_text_mask.sum()
    df = df[~empty_text_mask].copy()

    # Map categories
    df["sentiment"] = df["category"].apply(map_category_to_sentiment)
    invalid_cat_mask = df["sentiment"].isna()
    invalid_cat_count = invalid_cat_mask.sum()
    df = df[~invalid_cat_mask].copy()

    # Ensure original_category is typed as integer
    df["original_category"] = df["category"].astype(float).astype(int)

    # Drop duplicates
    duplicate_count = df.duplicated(subset=["tweet_text"]).sum()
    df = df.drop_duplicates(subset=["tweet_text"]).reset_index(drop=True)

    # Add useful analytical metadata columns
    df["tweet_id"] = [f"TW_{i + 1:06d}" for i in range(len(df))]
    df["char_length"] = df["tweet_text"].str.len()
    df["word_count"] = df["tweet_text"].str.split().str.len()

    # Reorder columns
    ordered_cols = [
        "tweet_id",
        "tweet_text",
        "original_category",
        "sentiment",
        "char_length",
        "word_count"
    ]
    df = df[ordered_cols]

    stats = {
        "raw_count": initial_count,
        "null_count": int(null_count),
        "empty_text_count": int(empty_count),
        "invalid_category_count": int(invalid_cat_count),
        "duplicate_count": int(duplicate_count),
        "final_valid_count": len(df)
    }

    return df, stats


def load_dataset(
    csv_path: Optional[str] = None,
    force_reload: bool = False,
    cache_path: str = LOCAL_CACHE_FILE
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads and returns the cleaned Twitter Sentiment Dataset.
    Uses local Parquet cache if available to prevent reprocessing on Streamlit reruns.

    Parameters:
        csv_path: Optional custom CSV path. If None, downloads via kagglehub.
        force_reload: If True, forces redownload and preprocessing.
        cache_path: Path to cached parquet file.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Processed DataFrame, Metadata Dictionary)
    """
    # Check if cached file exists
    if not force_reload and os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            metadata = {
                "source": "Local Parquet Cache",
                "file_path": cache_path,
                "total_rows": len(cached_df),
                "is_cached": True
            }
            return cached_df, metadata
        except Exception:
            # Fall back to reprocessing if cache read fails
            pass

    # Download from Kaggle if no path provided
    if csv_path is None or not os.path.exists(csv_path):
        csv_path = download_kaggle_dataset()

    raw_df = pd.read_csv(csv_path)
    processed_df, stats = preprocess_dataframe(raw_df)

    # Save to local cache directory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        processed_df.to_parquet(cache_path, index=False)
    except Exception:
        # Fall back to CSV cache if parquet engine is unavailable
        csv_cache = cache_path.replace(".parquet", ".csv")
        processed_df.to_csv(csv_cache, index=False)

    metadata = {
        "source": "Kaggle (saurabhshahane/twitter-sentiment-dataset)",
        "file_path": csv_path,
        "total_rows": len(processed_df),
        "cleaning_stats": stats,
        "is_cached": False
    }

    return processed_df, metadata


if __name__ == "__main__":
    print("Testing dataset_loader.py...")
    df, meta = load_dataset()
    print("Loaded dataset successfully.")
    print("Total valid records:", len(df))
    print("Sentiment distribution:")
    print(df["sentiment"].value_counts().to_string())
    print("\nPreview:")
    print(df.head(3).to_string())
