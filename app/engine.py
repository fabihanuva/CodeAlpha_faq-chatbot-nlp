import json
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple, Optional

class SemanticEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", faq_path: str = "data/faqs.json"):
        """
        Initialize the Semantic Search Engine.
        
        Args:
            model_name: The Sentence-Transformer model to use.
            faq_path: Path to the faqs.json file.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.faq_path = faq_path
        self.faqs: List[Dict] = []
        self.faq_embeddings: Optional[torch.Tensor] = None
        
        self.load_data()
        self.compute_embeddings()

    def load_data(self):
        """Load FAQs from the JSON file."""
        if not os.path.exists(self.faq_path):
            raise FileNotFoundError(f"FAQ file not found at {self.faq_path}")
        
        with open(self.faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

    def compute_embeddings(self):
        """Pre-compute embeddings for all FAQ questions."""
        questions = [faq["question"] for faq in self.faqs]
        self.faq_embeddings = self.model.encode(
            questions, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )

    def get_best_match(self, query: str, threshold: float = 0.4) -> Dict:
        """
        Find the best matching FAQ for a user query.
        
        Args:
            query: The user's input string.
            threshold: Minimum similarity score (0 to 1).
            
        Returns:
            A dictionary containing the match results.
        """
        if not query or not query.strip():
            return self._fallback_response("Please ask a question!")

        # Compute query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarities
        cos_scores = util.cos_sim(query_embedding, self.faq_embeddings)[0]
        
        # Get the highest score
        top_score, top_idx = torch.max(cos_scores, dim=0)
        score = float(top_score)
        idx = int(top_idx)

        if score < threshold:
            return self._fallback_response("I'm not quite sure about that. Could you rephrase?", score)

        matched_faq = self.faqs[idx]
        return {
            "answer": matched_faq["answer"],
            "score": score,
            "matched_question": matched_faq["question"],
            "category": matched_faq["category"],
            "matched": True
        }

    def _fallback_response(self, message: str, score: float = 0.0) -> Dict:
        return {
            "answer": message,
            "score": score,
            "matched_question": None,
            "category": None,
            "matched": False
        }

    def get_all_faqs(self) -> List[Dict]:
        """Return basic info for all FAQs (used for UI starter cards)."""
        return [{"id": f["id"], "question": f["question"], "category": f["category"]} for f in self.faqs]
