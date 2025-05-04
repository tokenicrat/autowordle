# Thank you again Claude

import json
from collections import Counter

def calculate_word_weights():
    # Load the word list
    with open('../worddb/parsed.json', 'r') as f:
        words = json.load(f)

    # Count character frequencies across all words
    char_counts = Counter(''.join(words))
    total_chars = sum(char_counts.values())

    # Calculate character weights (normalized frequencies)
    char_weights = {char: count/total_chars for char, count in char_counts.items()}

    # Calculate word weights using unique characters
    weighted_words = []
    for word in words:
        # Only count each unique character once per word
        unique_chars = set(word)
        weight = sum(char_weights[c] for c in unique_chars)
        weighted_words.append({"word": word, "weight": weight})

    # Sort by weight in descending order
    weighted_words.sort(key=lambda x: x["weight"], reverse=True)

    # Save the results
    with open('../worddb/weighted_words.json', 'w') as f:
        json.dump(weighted_words, f, indent=2)

if __name__ == "__main__":
    calculate_word_weights()