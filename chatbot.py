"""
chatbot.py
==========
FAQ Chatbot Engine — Task 2: NLP-Powered FAQ Chatbot
-----------------------------------------------------
Loads FAQ data, builds a TF-IDF model over preprocessed questions,
and matches user queries using cosine similarity.

Provides:
    - get_best_match(query)  → (answer, score, matched_question, category)
    - run_cli()              → interactive terminal chatbot loop

Usage:
    python chatbot.py
"""

import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAQ_FILE          = "faqs.json"
CONFIDENCE_THRESHOLD = 0.15   # Minimum score to return a match
BOT_NAME          = "ShopBot"

FALLBACK_RESPONSES = [
    "I'm sorry, I couldn't find a good answer for that. Could you rephrase your question?",
    "Hmm, I don't have information on that. Try asking about shipping, returns, payments, or your account.",
    "I'm not sure about that one. You can also reach our support team at support@store.com.",
]

WELCOME_BANNER = f"""
╔══════════════════════════════════════════════════════╗
║         Welcome to {BOT_NAME} — FAQ Assistant          ║
║                                                      ║
║  I can help you with:                                ║
║   • Shipping & Delivery                              ║
║   • Returns & Refunds                                ║
║   • Payments & Billing                               ║
║   • Order Management                                 ║
║   • Account & Profile                                ║
║                                                      ║
║  Type 'quit' or 'exit' to end the chat.              ║
║  Type 'help' to see all available topics.            ║
╚══════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Step 1 — Load FAQ data
# ---------------------------------------------------------------------------

def load_faqs(filepath: str) -> list[dict]:
    """
    Load FAQ entries from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the faqs.json file.

    Returns
    -------
    list[dict]
        List of FAQ dicts, each with keys: id, category, question, answer.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"FAQ file not found: '{filepath}'\n"
            f"Make sure faqs.json is in the same folder as chatbot.py."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        faqs = json.load(f)
    print(f"[{BOT_NAME}] Loaded {len(faqs)} FAQ entries from '{filepath}'.")
    return faqs


# ---------------------------------------------------------------------------
# Step 2 — Build TF-IDF model
# ---------------------------------------------------------------------------

def build_tfidf_model(faqs: list[dict]) -> tuple:
    """
    Preprocess all FAQ questions and fit a TF-IDF vectorizer.

    TF-IDF (Term Frequency–Inverse Document Frequency) converts text
    into numerical vectors. Words that appear often in one FAQ but rarely
    across all FAQs get higher weight — making them more discriminative.

    ngram_range=(1, 2) means we capture both single words AND two-word
    phrases (bigrams), e.g. "return policy", "shipping address". This
    significantly improves matching for multi-word concepts.

    Parameters
    ----------
    faqs : list[dict]
        Loaded FAQ entries.

    Returns
    -------
    tuple : (vectorizer, faq_matrix, preprocessed_questions)
        vectorizer          — fitted TfidfVectorizer
        faq_matrix          — (n_faqs × n_features) sparse matrix
        preprocessed_questions — list of cleaned FAQ question strings
    """
    # Preprocess every FAQ question
    preprocessed_questions = [preprocess(faq["question"]) for faq in faqs]

    # Fit TF-IDF vectorizer on the FAQ corpus
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=1,               # include even rare terms
        sublinear_tf=True,      # apply log normalization to TF
    )
    faq_matrix = vectorizer.fit_transform(preprocessed_questions)

    print(f"[{BOT_NAME}] TF-IDF model built.")
    print(f"[{BOT_NAME}] Vocabulary size: {len(vectorizer.vocabulary_)} terms.")
    return vectorizer, faq_matrix, preprocessed_questions


# ---------------------------------------------------------------------------
# Step 3 — Match user query
# ---------------------------------------------------------------------------

def get_best_match(
    user_query: str,
    faqs: list[dict],
    vectorizer: TfidfVectorizer,
    faq_matrix,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Find the best matching FAQ for a user query using cosine similarity.

    Cosine similarity measures the angle between two TF-IDF vectors.
    A score of 1.0 means identical, 0.0 means completely unrelated.
    We return the FAQ with the highest score above the threshold.

    Parameters
    ----------
    user_query : str
        Raw user input string.
    faqs : list[dict]
        Loaded FAQ dataset.
    vectorizer : TfidfVectorizer
        Fitted vectorizer (from build_tfidf_model).
    faq_matrix : sparse matrix
        TF-IDF matrix of all FAQ questions.
    threshold : float
        Minimum cosine similarity score to return a match.

    Returns
    -------
    dict with keys:
        answer          — the answer to return to the user
        score           — cosine similarity score (float, 0–1)
        matched_question — the FAQ question that matched
        category        — FAQ category of the matched entry
        matched         — True if above threshold, False if fallback
    """
    # Handle empty input
    if not user_query or not user_query.strip():
        return {
            "answer": "Please type a question and I'll do my best to help!",
            "score": 0.0,
            "matched_question": None,
            "category": None,
            "matched": False,
        }

    # Preprocess the user query
    cleaned_query = preprocess(user_query)

    # If preprocessing yields empty string (e.g. all stopwords)
    if not cleaned_query:
        return {
            "answer": FALLBACK_RESPONSES[0],
            "score": 0.0,
            "matched_question": None,
            "category": None,
            "matched": False,
        }

    # Transform query into TF-IDF vector
    query_vector = vectorizer.transform([cleaned_query])

    # Compute cosine similarity against all FAQ vectors
    scores = cosine_similarity(query_vector, faq_matrix).flatten()

    # Get the index of the highest scoring FAQ
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    # Apply confidence threshold
    if best_score < threshold:
        # Rotate through fallback responses based on query length (variety)
        fallback = FALLBACK_RESPONSES[len(user_query) % len(FALLBACK_RESPONSES)]
        return {
            "answer": fallback,
            "score": best_score,
            "matched_question": None,
            "category": None,
            "matched": False,
        }

    return {
        "answer": faqs[best_idx]["answer"],
        "score": best_score,
        "matched_question": faqs[best_idx]["question"],
        "category": faqs[best_idx]["category"],
        "matched": True,
    }


# ---------------------------------------------------------------------------
# Step 4 — CLI Chatbot Loop
# ---------------------------------------------------------------------------

def show_help(faqs: list[dict]) -> None:
    """Print all available FAQ questions grouped by category."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for faq in faqs:
        grouped[faq["category"]].append(faq["question"])

    print("\n" + "─" * 54)
    print("  Available Topics")
    print("─" * 54)
    for category, questions in grouped.items():
        print(f"\n  📂 {category}")
        for q in questions:
            print(f"     • {q}")
    print("─" * 54 + "\n")


def run_cli() -> None:
    """
    Interactive CLI chatbot loop.
    Loads FAQs, builds model, then accepts user input until 'quit'.
    """
    print(WELCOME_BANNER)

    # Load data and build model
    faqs = load_faqs(FAQ_FILE)
    vectorizer, faq_matrix, _ = build_tfidf_model(faqs)
    print(f"\n[{BOT_NAME}] Ready! Ask me anything.\n")
    print("─" * 54)

    # Chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{BOT_NAME}: Goodbye! Have a great day. 👋\n")
            break

        # Empty input
        if not user_input:
            continue

        # Exit commands
        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print(f"\n{BOT_NAME}: Goodbye! Have a great day. 👋\n")
            break

        # Help command
        if user_input.lower() == "help":
            show_help(faqs)
            continue

        # Get best match
        result = get_best_match(user_input, faqs, vectorizer, faq_matrix)

        # Display response
        print()
        if result["matched"]:
            print(f"  [{result['category']}]")
        print(f"{BOT_NAME}: {result['answer']}")

        # Debug info (confidence score) — comment out to hide
        if result["matched"]:
            print(f"\n  ↳ Matched: \"{result['matched_question']}\"")
            print(f"  ↳ Confidence: {result['score']:.2%}")
        else:
            print(f"\n  ↳ No confident match found. (score: {result['score']:.2%})")

        print("─" * 54)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_cli()