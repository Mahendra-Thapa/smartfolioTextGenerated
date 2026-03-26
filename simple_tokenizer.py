"""
simple_tokenizer.py
--------------------
Standalone module for SimpleTokenizer.
Must be a SEPARATE file so pickle can always find it
regardless of which script is running (app.py, train.py, lstm_model.py).
"""

class SimpleTokenizer:
    def __init__(self):
        self.word_index   = {}   # word  -> int (1-based)
        self.index_word   = {}   # int   -> word
        self._max_seq_len = 0

    def fit(self, sentences):
        freq = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                freq[word] = freq.get(word, 0) + 1
        for i, word in enumerate(sorted(freq, key=freq.get, reverse=True), start=1):
            self.word_index[word] = i
            self.index_word[i]    = word
        print(f"[Tokenizer] Vocabulary: {len(self.word_index)} words")

    def texts_to_sequences(self, texts):
        return [
            [self.word_index[w] for w in text.lower().split() if w in self.word_index]
            for text in texts
        ]

    @property
    def vocab_size(self):
        return len(self.word_index) + 1