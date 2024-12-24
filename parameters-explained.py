import numpy as np

# Vocabulary (expanded with non-relevant words)
vocab = ["I", "am", "happy", "to", "program", "this", "and", "learn", "chicken", "banking", "Ronaldo"]

# Assigning weights (importance) to words
weights = {
    "I": np.array([0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01]),     # High weight for "am"
    "am": np.array([0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01]),   # High weight for "happy"
    "happy": np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01]), # High weight for "to"
    "to": np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01]),   # High weight for "program"
    "program": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.01, 0.01, 0.01]), # High weight for "this"
    "this": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.01, 0.01, 0.01]), # High weight for "and"
    "and": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.01, 0.01, 0.01]),  # High weight for "learn"
    "learn": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01]), # End of sentence
    "chicken": np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    "banking": np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    "Ronaldo": np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
}

# Biases (to ensure predictions stay within reasonable bounds)
biases = np.array([0.1, 0.1, -0.5, 0.1, 0.1, 0.1, 0.1, 0.1, -0.9, -0.9, -0.9])  # Lower bias for irrelevant words

def predict_next_word(current_word):
    """
    Predicts the next word based on weights and biases.
    
    Args:
        current_word (str): The current word in the sequence.
    
    Returns:
        str: The predicted next word.
    """
    if current_word not in weights:
        return None  # Return None if the word is not in the vocabulary

    # Fetch weights for the current word
    word_weights = weights[current_word]

    # Apply biases to the weights
    adjusted_scores = word_weights + biases

    # Use softmax to convert scores to probabilities
    probabilities = np.exp(adjusted_scores) / np.sum(np.exp(adjusted_scores))

    # Select the word with the highest probability
    next_word_index = np.argmax(probabilities)
    return vocab[next_word_index]

def generate_sentence(start_word, max_length=10):
    """
    Generates a sentence starting from a given word.
    
    Args:
        start_word (str): The first word of the sentence.
        max_length (int): The maximum length of the generated sentence.
    
    Returns:
        str: The generated sentence.
    """
    sentence = [start_word]
    current_word = start_word

    for _ in range(max_length - 1):  # Generate up to max_length words
        next_word = predict_next_word(current_word)
        if next_word is None or next_word == "learn":  # End sentence at "learn"
            break
        sentence.append(next_word)
        current_word = next_word

    return " ".join(sentence)

# Generate and print a sentence
start_word = "I"
generated_sentence = generate_sentence(start_word)
print(f"Generated Sentence: {generated_sentence}")
