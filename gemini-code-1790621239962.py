import math
import re
from collections import Counter

def extract_keywords_pnt(text: str, top_n: int = 5) -> list[tuple[str, float]]:
    # Tokenize text into lowercased words
    words = re.findall(r'\b[a-zA-L0-9]+\b', text.lower())
    if not words:
        return []

    total_words = len(words)
    term_counts = Counter(words)

    # Sort words by frequency (descending) to establish rank n
    ranked_words = term_counts.most_common()

    pnt_scores = {}
    for rank, (word, count) in enumerate(ranked_words, start=1):
        # Prevent log(1) division by zero for rank 1
        adjusted_rank = max(rank, 2)
        
        # PNT-inspired weight: inverse of prime density local factor (ln(n))
        # scaled by term frequency (TF) relative to text length
        pnt_density_weight = math.log(adjusted_rank)
        tf = count / total_words
        
        # Words appearing with high rank (larger rank index n) and high local TF score higher
        score = tf * pnt_density_weight
        
        # Filter out extremely short non-essential tokens
        if len(word) > 2:
            pnt_scores[word] = round(score, 4)

    # Return top N keywords sorted by PNT weight
    sorted_keywords = sorted(pnt_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]


# Example Usage:
if __name__ == "__main__":
    sample_sentence = (
        "Quantum computing leverages the principles of quantum mechanics "
        "to solve complex computational problems faster than classical supercomputers."
    )
    
    keywords = extract_keywords_pnt(sample_sentence, top_n=4)
    print("Extracted Keywords (Word, PNT Score):")
    for word, score in keywords:
        print(f" - {word}: {score}")