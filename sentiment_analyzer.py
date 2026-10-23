"""
sentiment_analyzer.py
=====================
Modular sentiment analysis module for the 'Twitter Sentiment Analysis Using Big Data Analytics' project.

Technology:
- TextBlob (Lexicon & rule-based sentiment calculation via PatternAnalyzer)
- Regular Expressions for tweet text normalization and preprocessing

Design & Methodology:
1. Text Cleaning: Strips URLs, user handles (@username), hashtags (retaining keyword),
   and abnormal whitespace while preserving semantic punctuation.
2. Polarity Scoring: Calculates a float score in [-1.0, 1.0].
   - Values > pos_threshold (default 0.1) are classified as 'Positive'.
   - Values < neg_threshold (default -0.1) are classified as 'Negative'.
   - Values in between [-0.1, 0.1] are classified as 'Neutral'.
3. Subjectivity Scoring: Calculates a float score in [0.0, 1.0] representing factual vs. opinionated.
4. Edge Case Handling: Robust against None, empty strings, pure whitespace, numeric types,
   and malformed inputs.

Limitations Note:
- TextBlob uses an underlying lexicon (Pattern library) calibrated on general English text.
- It has limited awareness of nuanced Internet slang, sarcastic remarks, complex emoji combinations,
  and context-dependent irony. These limitations are intentional trade-offs for computational
  simplicity, zero-credential local execution, and academic explainability.
"""

import re
from typing import Dict, Any, List, Union
import pandas as pd
from textblob import TextBlob


# Default configurable polarity thresholds
DEFAULT_POS_THRESHOLD = 0.1
DEFAULT_NEG_THRESHOLD = -0.1


def clean_tweet(text: Any) -> str:
    """
    Cleans and normalizes tweet text for NLP processing.

    Cleaning steps:
    1. Null/type validation: Converts non-string/null input to empty string.
    2. URL removal: Removes 'http://', 'https://', and 'www.' links.
    3. Mentions removal: Removes user handles (e.g. '@user123').
    4. Retweet marker removal: Removes 'RT ' prefixes.
    5. Hashtag symbol normalization: Removes the '#' symbol while retaining the tag word.
    6. Whitespace collapse: Normalizes multiple spaces, tabs, and newlines into single spaces.

    Parameters:
        text (Any): Raw tweet text or object.

    Returns:
        str: Cleaned, trimmed tweet text.
    """
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except Exception:
        pass

    text_str = str(text).strip()
    if not text_str or text_str in ["[]", "{}"]:
        return ""

    # Remove standard Retweet prefix: 'RT @user: '
    text_str = re.sub(r"^RT[\s]+", "", text_str)

    # Remove URLs (http, https, www)
    text_str = re.sub(r"https?://\S+|www\.\S+", "", text_str)

    # Remove user mentions (@username)
    text_str = re.sub(r"@\w+", "", text_str)

    # Normalize hashtags: retain the word without the '#' symbol
    text_str = re.sub(r"#(\w+)", r"\1", text_str)

    # Remove HTML entities like &amp;, &lt;, &gt;
    text_str = re.sub(r"&\w+;", " ", text_str)

    # Normalize extra whitespace
    text_str = re.sub(r"\s+", " ", text_str).strip()

    return text_str


def analyze_sentiment(
    text: Any,
    pos_threshold: float = DEFAULT_POS_THRESHOLD,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD
) -> Dict[str, Any]:
    """
    Calculates polarity, subjectivity, and predicted sentiment label for a single tweet.

    Parameters:
        text (Any): Raw or semi-cleaned text input.
        pos_threshold (float): Minimum polarity score required to qualify as 'Positive'.
        neg_threshold (float): Maximum polarity score required to qualify as 'Negative'.

    Returns:
        Dict[str, Any]: {
            'cleaned_text': str,
            'polarity': float (rounded to 4 decimals, range [-1.0, 1.0]),
            'subjectivity': float (rounded to 4 decimals, range [0.0, 1.0]),
            'sentiment': 'Positive' | 'Negative' | 'Neutral',
            'explanation': str,
            'model_note': str
        }
    """
    cleaned = clean_tweet(text)

    model_note = (
        "Classification is a model prediction based on lexical polarity and "
        "may not correctly interpret sarcasm, internet slang, or contextual language."
    )

    # Safeguard against empty or whitespace-only text
    if not cleaned:
        return {
            "cleaned_text": "",
            "polarity": 0.0,
            "subjectivity": 0.0,
            "sentiment": "Neutral",
            "explanation": "No valid text detected after cleaning; classified as Neutral by default.",
            "model_note": model_note
        }

    try:
        blob = TextBlob(cleaned)
        polarity = round(float(blob.sentiment.polarity), 4)
        subjectivity = round(float(blob.sentiment.subjectivity), 4)
    except Exception:
        polarity = 0.0
        subjectivity = 0.0

    # Categorization based on configurable thresholds
    if polarity > pos_threshold:
        sentiment = "Positive"
        explanation = (
            f"Polarity score is {polarity:+.4f}. Since this is greater than the positive threshold "
            f"({pos_threshold:+.2f}), the text is classified as Positive."
        )
    elif polarity < neg_threshold:
        sentiment = "Negative"
        explanation = (
            f"Polarity score is {polarity:+.4f}. Since this is less than the negative threshold "
            f"({neg_threshold:+.2f}), the text is classified as Negative."
        )
    else:
        sentiment = "Neutral"
        explanation = (
            f"Polarity score is {polarity:+.4f}. Since this falls within the neutral threshold window "
            f"[{neg_threshold:+.2f}, {pos_threshold:+.2f}], the text is classified as Neutral."
        )

    return {
        "cleaned_text": cleaned,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment,
        "explanation": explanation,
        "model_note": model_note
    }


def batch_analyze_dataframe(
    df: pd.DataFrame,
    text_column: str = "tweet_text",
    pos_threshold: float = DEFAULT_POS_THRESHOLD,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD
) -> pd.DataFrame:
    """
    Applies sentiment analysis across an entire Pandas DataFrame.

    Parameters:
        df (pd.DataFrame): Input dataframe containing tweet records.
        text_column (str): Name of the column containing tweet texts.
        pos_threshold (float): Positive threshold limit.
        neg_threshold (float): Negative threshold limit.

    Returns:
        pd.DataFrame: A copy of the input DataFrame augmented with
                      'cleaned_text', 'polarity', 'subjectivity', and 'predicted_sentiment'.
    """
    if text_column not in df.columns:
        raise KeyError(f"Specified text column '{text_column}' not found in DataFrame columns: {list(df.columns)}")

    result_df = df.copy()

    # Pre-clean texts
    cleaned_texts = [clean_tweet(t) for t in result_df[text_column]]
    result_df["cleaned_text"] = cleaned_texts

    # Compute sentiment scores
    polarities = []
    subjectivities = []
    sentiments = []

    for text in cleaned_texts:
        if not text:
            polarities.append(0.0)
            subjectivities.append(0.0)
            sentiments.append("Neutral")
            continue

        try:
            blob = TextBlob(text)
            pol = round(float(blob.sentiment.polarity), 4)
            sub = round(float(blob.sentiment.subjectivity), 4)
        except Exception:
            pol = 0.0
            sub = 0.0

        polarities.append(pol)
        subjectivities.append(sub)

        if pol > pos_threshold:
            sentiments.append("Positive")
        elif pol < neg_threshold:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

    result_df["polarity"] = polarities
    result_df["subjectivity"] = subjectivities
    result_df["predicted_sentiment"] = sentiments
    result_df["sentiment"] = sentiments

    return result_df
