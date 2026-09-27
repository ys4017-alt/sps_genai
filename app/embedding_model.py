"""Word embedding functionality ported from Module 2 Practical 3.

The notebook used spaCy's large English model, which ships 300-dimensional
word vectors, and exposed two functions:

    calculate_embedding(input_word)      -> nlp(input_word).vector
    calculate_similarity(word1, word2)   -> nlp(word1).similarity(nlp(word2))

Both are wrapped in a class here so the model is loaded once at import time
instead of on every request.
"""

import spacy

# en_core_web_lg is required, not en_core_web_sm: the small model ships zero
# word vectors and silently falls back to context tensors, which makes
# similarity scores meaningless (apple/car scores 0.73 instead of 0.22).
MODEL_NAME = "en_core_web_lg"


class EmbeddingModel:
    """Wraps the Module 2 Practical 3 embedding functions for the API."""

    def __init__(self, model_name=MODEL_NAME):
        self.nlp = spacy.load(model_name)
        self.model_name = model_name
        self.dimensions = self.nlp.vocab.vectors.shape[1]

    def calculate_embedding(self, input_word):
        """Return the embedding vector for a word as a plain list of floats."""
        word = self.nlp(input_word)
        return word.vector.tolist()

    def calculate_similarity(self, word1, word2):
        """Cosine similarity between two words (or two pieces of text)."""
        return float(self.nlp(word1).similarity(self.nlp(word2)))

    def has_vector(self, text):
        """True when every token of `text` is in the model's vector table.

        Out-of-vocabulary words get a zero vector, which makes similarity 0
        without raising, so the API reports this instead of failing silently.
        """
        doc = self.nlp(text)
        return bool(len(doc)) and all(token.has_vector for token in doc)
