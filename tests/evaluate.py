import json
import os
import sys
from typing import List, Dict

# Add parent directory to path to allow importing app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import SemanticEngine

FAQ_FILE = "data/faqs.json"

TEST_CASES = [
    {"query": "When will my package arrive?", "expected_id": 1},
    {"query": "Do you deliver to other countries?", "expected_id": 4},
    {"query": "Can I get free delivery on my order?", "expected_id": 5},
    {"query": "My order hasn't shown up yet, what now?", "expected_id": 6},
    {"query": "I need to update the delivery address on my order.", "expected_id": 7},
    {"query": "How do I send something back?", "expected_id": 10},
    {"query": "How many days until I get my money back?", "expected_id": 11},
    {"query": "I got the wrong item in my delivery.", "expected_id": 14},
    {"query": "What cards do you accept for payment?", "expected_id": 16},
    {"query": "My card was rejected at checkout.", "expected_id": 19},
    {"query": "I have a promo code, where do I enter it?", "expected_id": 21},
    {"query": "I want to stop my order from being sent.", "expected_id": 22},
    {"query": "I never got a confirmation email for my purchase.", "expected_id": 25},
    {"query": "I need to reset my login password.", "expected_id": 28},
    {"query": "How do I permanently delete my profile?", "expected_id": 30},
]

def run_evaluation():
    print("\n" + "=" * 70)
    print("  SHOPBOT SEMANTIC ENGINE — EVALUATION")
    print("=" * 70)
    
    engine = SemanticEngine(faq_path=FAQ_FILE)
    
    # Load FAQ lookup
    with open(FAQ_FILE, "r") as f:
        faqs = json.load(f)
    faq_by_id = {f["id"]: f for f in faqs}

    passed = 0
    for i, test in enumerate(TEST_CASES, start=1):
        query = test["query"]
        expected_faq = faq_by_id[test["expected_id"]]
        
        result = engine.get_best_match(query)
        is_correct = result["answer"] == expected_faq["answer"]
        
        status = "PASS ✓" if is_correct else "FAIL ✗"
        if is_correct: passed += 1
        
        print(f"[{i:02d}] {status} | Score: {result['score']:.2%} | Query: {query}")
        if not is_correct:
            print(f"    Expected: {expected_faq['question']}")
            print(f"    Got:      {result['matched_question']}")

    accuracy = (passed / len(TEST_CASES)) * 100
    print(f"\nFINAL ACCURACY: {accuracy:.1f}% ({passed}/{len(TEST_CASES)})")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_evaluation()
