# 🛍️ ShopBot — NLP-Powered FAQ Chatbot

A smart FAQ chatbot that understands natural language questions and matches them to the most relevant answer using classic NLP techniques — **TF-IDF vectorization** and **cosine similarity**. No external APIs, no LLMs, fully offline.

Built as part of an NLP practicum to demonstrate end-to-end text processing: from raw text to an interactive chat interface.

---

## ✨ Demo

Open `chatbot_ui.html` in any browser — no server required.

**Try asking:**
- "When will my package arrive?"
- "My card got rejected"
- "How do I send something back?"
- "I forgot my login password"

The bot matches your question to the closest FAQ — even when worded completely differently — and highlights the exact words that triggered the match.

---

## 🧠 How It Works

```
User Question
     │
     ▼
┌─────────────────────────┐
│  1. Preprocessing        │   lowercase → expand contractions →
│                           │   remove punctuation → tokenize →
│                           │   remove stopwords → lemmatize
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  2. TF-IDF Vectorization │   Converts text into numerical vectors
│                           │   based on word importance (unigrams + bigrams)
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  3. Cosine Similarity    │   Compares the question vector against
│                           │   all 34 FAQ vectors → finds best match
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  4. Confidence Threshold │   If similarity ≥ 0.10 → return FAQ answer
│                           │   If below → return a fallback response
└─────────────────────────┘
     │
     ▼
   Chat Response
```

---

## 📂 Project Structure

```
faq_chatbot/
├── faqs.json          # 34 FAQ pairs across 5 categories
├── preprocess.py       # NLP preprocessing pipeline (NLTK)
├── chatbot.py           # TF-IDF + cosine similarity engine + CLI
├── evaluate.py           # Accuracy evaluation on 15 paraphrased queries
├── chatbot_ui.html        # Standalone interactive chat UI
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### Option A — Just try the UI (fastest)
Download `chatbot_ui.html` and open it in your browser. That's it — the NLP engine runs entirely in JavaScript.

### Option B — Run the Python version

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/faq-chatbot-nlp.git
cd faq-chatbot-nlp
```

**2. Set up a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download NLTK data** (one-time)
```bash
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**5. Run the chatbot**
```bash
python chatbot.py
```

---

## 📊 Evaluation Results

Tested against 15 paraphrased queries — questions worded completely differently from the original FAQs.

```bash
python evaluate.py
```

| Category | Accuracy |
|---|---|
| Shipping & Delivery | 100% |
| Returns & Refunds | 100% |
| Payments & Billing | 100% |
| Order Management | 100% |
| Account & Profile | 100% |
| **Overall** | **100% (15/15)** |

---

## 🎨 Chat UI Features

- **Keyword highlighting** — shows exactly which words in your question matched the FAQ
- **Follow-up suggestions** — smart, category-based next questions after each answer
- **Dark mode** — full theme toggle with smooth transitions
- **Starter cards** — quick-access topics on first load
- **Typing indicator** — natural conversational feel
- **Fully responsive** — works on mobile and desktop

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **NLTK** — tokenization, stopwords, lemmatization
- **scikit-learn** — TF-IDF vectorization, cosine similarity
- **NumPy**
- **HTML / CSS / Vanilla JS** — chat interface (mirrors the Python logic)

---

## 📌 Future Improvements

- Add fuzzy matching for typo tolerance
- Expand FAQ dataset with more categories
- Add multi-turn conversation context
- Deploy as a web app with a small backend API

---

## 📄 License

MIT License — feel free to use this project as a learning reference or starting point for your own FAQ chatbot.