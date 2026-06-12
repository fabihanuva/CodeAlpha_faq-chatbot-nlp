"""
preprocess.py
=============
NLP Preprocessing Pipeline — Task 2: FAQ Chatbot
-------------------------------------------------
Reusable text preprocessing module built on NLTK.
Provides a single clean interface: preprocess(text) -> str

Pipeline steps:
    1. Lowercase
    2. Remove punctuation & special characters
    3. Tokenize (word-level)
    4. Remove stopwords
    5. Lemmatize

Usage:
    from preprocess import preprocess
    clean = preprocess("How long does standard shipping take?")
    # → "long standard shipping take"

Setup (run once before using this module):
    pip install nltk
    python3 -c "import nltk; nltk.download('all')"
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# One-time NLTK data download guard
# ---------------------------------------------------------------------------

def _download_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    required = {
        "punkt_tab":  "tokenizers/punkt_tab",
        "stopwords":  "corpora/stopwords",
        "wordnet":    "corpora/wordnet",
        "omw-1.4":    "corpora/omw-1.4",
    }
    for name, path in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[preprocess] Downloading NLTK data: '{name}' ...")
            nltk.download(name, quiet=True)


_download_nltk_data()


# ---------------------------------------------------------------------------
# Module-level singletons (created once, reused across all calls)
# ---------------------------------------------------------------------------

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Words that carry meaning for FAQ matching — remove from stopword list
# e.g. "not" in "item not arrived" matters for intent
_STOPWORD_EXCEPTIONS = {"not", "no", "never", "free", "wrong", "what", "why", "where", "who"}
_stop_words -= _STOPWORD_EXCEPTIONS


# ---------------------------------------------------------------------------
# Core preprocessing steps (each step is a pure function)
# ---------------------------------------------------------------------------

def to_lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def expand_contractions(text: str) -> str:
    """
    Expand common English contractions before tokenization.
    e.g. "haven't" → "have not", "don't" → "do not", "I'm" → "I am"
    This ensures "not" is preserved as a meaningful token.
    """
    contractions = {
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "cannot",
        "cannot": "cannot",
        "couldn't": "could not",
        "shouldn't": "should not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "i'm": "i am",
        "i've": "i have",
        "i'll": "i will",
        "i'd": "i would",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "they're": "they are",
        "they've": "they have",
        "we're": "we are",
        "we've": "we have",
        "you're": "you are",
        "you've": "you have",
        "what's": "what is",
        "who's": "who is",
        "let's": "let us",
    }
    # Use word-boundary regex for accurate replacement
    for contraction, expansion in contractions.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text)
    return text


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation and special characters.
    Keeps only letters, numbers, and spaces.
    Called AFTER expand_contractions so apostrophes are already resolved.
    """
    # Replace any non-alphanumeric character (except space) with a space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split text into individual word tokens using NLTK word_tokenize."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove common English stopwords.
    Preserves exception words that carry FAQ-matching intent.
    Also drops pure numeric tokens (e.g. '50', '7').
    """
    return [
        token for token in tokens
        if token not in _stop_words and not token.isnumeric()
    ]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Reduce each token to its base (lemma) form.
    e.g. 'shipping' → 'shipping', 'orders' → 'order', 'received' → 'receive'
    Tries verb form first, falls back to noun form (NLTK default).
    """
    return [_lemmatizer.lemmatize(token, pos="v") for token in tokens]


# ---------------------------------------------------------------------------
# Main public interface
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Steps applied in order:
        1. Lowercase
        2. Remove punctuation / special characters
        3. Tokenize
        4. Remove stopwords (with intent-preserving exceptions)
        5. Lemmatize

    Parameters
    ----------
    text : str
        Raw input string (user query or FAQ question).

    Returns
    -------
    str
        Single whitespace-joined string of cleaned, lemmatized tokens.
        Returns empty string if input is empty or produces no tokens.

    Examples
    --------
    >>> preprocess("How long does standard shipping take?")
    'long standard shipping take'

    >>> preprocess("I haven't received my order yet — what should I do?")
    'not receive order yet'

    >>> preprocess("Can I return a damaged item?")
    'return damage item'
    """
    if not text or not text.strip():
        return ""

    text = to_lowercase(text)
    text = expand_contractions(text)   # "haven't" → "have not" BEFORE punctuation removal
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """
    5 unit tests covering core preprocessing behaviour.
    Prints PASS / FAIL for each test case.
    """
    print("=" * 55)
    print("  preprocess.py — Unit Tests")
    print("=" * 55)

    test_cases = [
        # (test_name, input_text, expected_output)
        (
            "T1: Basic question",
            "How long does standard shipping take?",
            "how long standard ship take",
        ),
        (
            "T2: Punctuation & special chars",
            "What's your return policy — any exceptions?",
            "what return policy exceptions",
        ),
        (
            "T3: Contraction handling",
            "I haven't received my order yet.",
            "not receive order yet",
        ),
        (
            "T4: Uppercase + stopwords",
            "CAN I CANCEL MY ORDER AFTER PLACING IT?",
            "cancel order place",
        ),
        (
            "T5: Empty string",
            "",
            "",
        ),
    ]

    passed = 0
    for name, input_text, expected in test_cases:
        result = preprocess(input_text)
        status = "PASS ✓" if result == expected else "FAIL ✗"
        if result == expected:
            passed += 1
        print(f"\n  {status}  {name}")
        print(f"    Input   : {repr(input_text)}")
        print(f"    Expected: {repr(expected)}")
        print(f"    Got     : {repr(result)}")

    print("\n" + "=" * 55)
    print(f"  Results: {passed}/{len(test_cases)} tests passed")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Demo: Run when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tests()

    print("\n--- Interactive Demo ---")
    print("Type a sentence to see it preprocessed. Press Ctrl+C to quit.\n")
    try:
        while True:
            raw = input("Input  : ")
            clean = preprocess(raw)
            print(f"Output : {repr(clean)}\n")
    except KeyboardInterrupt:
        print("\nDone.")