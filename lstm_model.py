"""
lstm_model.py
-------------
LSTM next-word prediction model.
SimpleTokenizer is imported from simple_tokenizer.py (standalone module)
so pickle always finds it correctly regardless of entry point.
"""

import os
import sys
import pickle
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus           import CORPUS
from simple_tokenizer import SimpleTokenizer   # ← standalone module, pickle safe

MODEL_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
MODEL_PATH     = os.path.join(MODEL_DIR, "lstm.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

EPOCHS     = 60
BATCH_SIZE = 64
EMBED_DIM  = 128
LSTM_UNITS = 256
TOP_K      = 5
NUM_WORDS  = 5

_model       = None
_tokenizer   = None
_max_seq_len = None
_ready       = threading.Event()


def _get_keras():
    import tensorflow as tf
    return tf.keras


def _pad_sequences(sequences, maxlen, padding="pre"):
    result = []
    for seq in sequences:
        seq = list(seq)
        if len(seq) < maxlen:
            pad = [0] * (maxlen - len(seq))
            seq = pad + seq if padding == "pre" else seq + pad
        else:
            seq = seq[-maxlen:]
        result.append(seq)
    return np.array(result, dtype=np.int32)


def _build_sequences(tokenizer, sentences):
    all_seqs = []
    for sentence in sentences:
        tokens = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(tokens)):
            all_seqs.append(tokens[: i + 1])
    max_len = max(len(s) for s in all_seqs)
    padded  = _pad_sequences(all_seqs, maxlen=max_len, padding="pre")
    return padded, max_len


def _build_model(vocab_size, seq_len):
    keras = _get_keras()
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, EMBED_DIM, input_length=seq_len - 1),
        keras.layers.LSTM(LSTM_UNITS, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(LSTM_UNITS),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train(extra_sentences=None):
    global _model, _tokenizer, _max_seq_len

    keras     = _get_keras()
    sentences = CORPUS[:]
    if extra_sentences:
        sentences.extend(extra_sentences)

    print(f"[LSTM] Training on {len(sentences)} sentences …")

    tokenizer = SimpleTokenizer()
    tokenizer.fit(sentences)
    vocab_size = tokenizer.vocab_size

    sequences, max_seq_len = _build_sequences(tokenizer, sentences)
    X = sequences[:, :-1]
    y = keras.utils.to_categorical(sequences[:, -1], num_classes=vocab_size)

    print(f"[LSTM] Samples={len(X)}, seq_len={max_seq_len}, vocab={vocab_size}")
    model = _build_model(vocab_size, max_seq_len)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    tokenizer._max_seq_len = max_seq_len
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    _model       = model
    _tokenizer   = tokenizer
    _max_seq_len = max_seq_len

    print(f"[LSTM] Training complete. Saved → {MODEL_PATH}")
    _ready.set()


def _load_saved():
    global _model, _tokenizer, _max_seq_len

    keras  = _get_keras()
    _model = keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)

    _max_seq_len = getattr(_tokenizer, "_max_seq_len", 50)
    print(f"[LSTM] Loaded. vocab={_tokenizer.vocab_size}, max_seq_len={_max_seq_len}")
    _ready.set()


def _startup():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
            print("[LSTM] Saved model found — loading …")
            _load_saved()
        else:
            print("[LSTM] No saved model — training from scratch …")
            train()
    except Exception as e:
        print(f"[LSTM] Startup error: {e}")
        import traceback; traceback.print_exc()
        _ready.set()


threading.Thread(target=_startup, daemon=True).start()


# ════════════════════════════════════════════════════════════════════
# Prediction
# ════════════════════════════════════════════════════════════════════

def _get_known_tokens(seed_text: str) -> list:
    words   = seed_text.strip().lower().split()
    tokens  = [_tokenizer.word_index[w] for w in words if w in _tokenizer.word_index]
    unknown = [w for w in words if w not in _tokenizer.word_index]

    if unknown:
        print(f"[LSTM] Unknown words: {unknown}")
    print(f"[LSTM] Known tokens: {tokens}")

    # fallback — use top-3 most frequent words (i, am, a)
    if not tokens:
        print("[LSTM] Fallback to top-3 vocab words")
        tokens = [1, 2, 3]

    return tokens


def _predict_one(seed_tokens: list, num_words: int, temperature: float = 1.0) -> str:
    tokens = list(seed_tokens)
    result = []

    for _ in range(num_words):
        padded = _pad_sequences([tokens], maxlen=_max_seq_len - 1, padding="pre")
        probs  = _model.predict(padded, verbose=0)[0].astype("float64")

        # temperature scaling
        probs = np.log(np.clip(probs, 1e-10, None)) / max(temperature, 0.1)
        probs = np.exp(probs - np.max(probs))
        probs = probs / probs.sum()

        top_idx   = np.argsort(probs)[-10:]
        top_probs = probs[top_idx]
        top_probs = top_probs / top_probs.sum()

        chosen = int(np.random.choice(top_idx, p=top_probs))
        word   = _tokenizer.index_word.get(chosen)

        if not word:
            break

        result.append(word)
        tokens.append(chosen)

    return " ".join(result)


def predict(seed_text: str, top_k: int = TOP_K, num_words: int = NUM_WORDS) -> list:
    if not seed_text.strip():
        return []

    _ready.wait(timeout=300)

    if _model is None or _tokenizer is None:
        print("[LSTM] Model not loaded")
        return []

    print(f"[LSTM] predict() seed='{seed_text}'")

    try:
        seed_tokens = _get_known_tokens(seed_text)
        suggestions = []
        seen        = set()
        temperatures = [0.4, 0.6, 0.8, 1.0, 1.2, 0.5, 0.7, 0.9, 1.1, 1.3]

        for temp in temperatures:
            if len(suggestions) >= top_k:
                break
            completion = _predict_one(seed_tokens, num_words, temperature=temp)
            if completion and completion not in seen:
                seen.add(completion)
                suggestions.append(completion)
                print(f"[LSTM] suggestion {len(suggestions)}: '{completion}'")

        print(f"[LSTM] returning {len(suggestions)} suggestions")
        return suggestions[:top_k]

    except Exception as e:
        print(f"[LSTM] predict error: {e}")
        import traceback; traceback.print_exc()
        return []


def retrain(extra_sentences=None):
    global _ready
    _ready.clear()
    threading.Thread(
        target=train,
        kwargs={"extra_sentences": extra_sentences},
        daemon=True,
    ).start()