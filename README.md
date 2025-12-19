# FinBERT-Long-Text: Smart Financial Sentiment Analysis

A Python wrapper for **FinBERT** designed to handle **long financial documents** (news articles, earnings calls, reports) that exceed the standard BERT 512-token limit. 

Unlike standard chunking methods that average sentiment (diluting the signal), this tool uses a **"Strongest Signal" (Max-Pooling)** strategy to detect the most significant sentiment in a document, ensuring critical news isn't washed out by neutral filler text.

## 🚀 Features

* **Automatic Chunking:** Splits long text into 510-token overlapping chunks automatically.
* **Strongest Signal Strategy:** Instead of averaging scores (which returns "Neutral" for mixed documents), it identifies the chunk with the highest Positive or Negative confidence.
* **GPU Acceleration:** Automatically uses NVIDIA CUDA if available.
* **Production Ready:** Handles special tokens (`[CLS]`, `[SEP]`) correctly for every chunk.

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/FinBERT-Long-Text.git](https://github.com/yourusername/FinBERT-Long-Text.git)
    cd FinBERT-Long-Text
    ```

2.  Install dependencies:
    ```bash
    pip install torch transformers numpy
    ```

## ⚡ Usage

Ensure `sentiment_analyzer.py` is in your project directory.

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize (Downloads model on first run)
analyzer = SentimentAnalyzer()

# Example: Long text with mixed history but positive recent news
text = """
    (Paragraph 1) Company X was founded in 1905... [Neutral History]
    (Paragraph 2) In 2010, they faced a minor lawsuit... [Negative Noise]
    (Paragraph 3) TODAY, they reported a record 50% profit increase! [Positive Signal]
"""

# The analyzer will detect the Positive signal despite the neutral/negative filler
confidence, sentiment = analyzer.analyze(text)

print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
# Output: Sentiment: Positive (Confidence: 0.98)
