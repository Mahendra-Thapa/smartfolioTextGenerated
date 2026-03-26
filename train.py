"""
train.py
--------
Proper standalone training script for the Portfolio LSTM model.

Trains ONE model on ALL 8 fields combined for best cross-field
suggestion quality. Uses callbacks for early stopping, best model
saving, and learning rate reduction.

Run this after updating corpus.py with your own data:
    python train.py

What it does:
  1. Loads all sentences from corpus.py
  2. Reports data stats per section
  3. Builds a word-level tokenizer
  4. Trains LSTM with validation, early stopping, best model saving
  5. Tests predictions on all 8 fields so you can see quality
"""

import os
import sys
import pickle
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import CORPUS

# ── Config ────────────────────────────────────────────────────────────────
EPOCHS           = 60
BATCH_SIZE       = 64
EMBED_DIM        = 128
LSTM_UNITS       = 256
VALIDATION_SPLIT = 0.1
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_PATH       = os.path.join(MODEL_DIR, "lstm.keras")
TOKENIZER_PATH   = os.path.join(MODEL_DIR, "tokenizer.pkl")


# ── Helpers ───────────────────────────────────────────────────────────────

def pad_sequences(sequences, maxlen, padding="pre"):
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


# SimpleTokenizer moved to simple_tokenizer.py
from simple_tokenizer import SimpleTokenizer
if False:  # dummy
 from simple_tokenizer import SimpleTokenizer



# ── Predict helper (used in test after training) ──────────────────────────

def test_predict(model, tokenizer, max_seq_len, seed, n_words=5):
    text   = seed.lower()
    result = []
    for _ in range(n_words):
        tokens = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([tokens], maxlen=max_seq_len - 1, padding="pre")
        probs  = model.predict(padded, verbose=0)[0]
        top5   = np.argsort(probs)[-5:][::-1]
        chosen = top5[0]
        word   = tokenizer.index_word.get(int(chosen), "")
        if not word:
            break
        result.append(word)
        text += " " + word
    return " ".join(result)


# ── Main ──────────────────────────────────────────────────────────────────

def train():
    import tensorflow as tf
    keras = tf.keras

    print("\n" + "=" * 60)
    print("  SmartFolioAI — LSTM Training Script")
    print("  Fields: Introduction, Bio, Skills, Projects,")
    print("          Experience, Additional Experience,")
    print("          Qualifications, Education")
    print("=" * 60)

    # ── 1. Load & report data ─────────────────────────────────────────
    sentences = CORPUS[:]
    print(f"\n[1/5] Data Summary")
    print(f"  Total sentences : {len(sentences)}")
    print(f"  Total words     : {sum(len(s.split()) for s in sentences)}")
    print(f"  Avg words/sent  : {sum(len(s.split()) for s in sentences) // len(sentences)}")

    if len(sentences) < 50:
        print(f"\n  WARNING: Only {len(sentences)} sentences.")
        print(f"  Add more to corpus.py for better quality. Aim for 150+.")

    # ── 2. Tokenize ───────────────────────────────────────────────────
    print(f"\n[2/5] Building tokenizer …")
    tokenizer = SimpleTokenizer()
    tokenizer.fit(sentences)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size : {vocab_size} unique words")

    # ── 3. Build sequences ────────────────────────────────────────────
    print(f"\n[3/5] Building training sequences …")
    all_sequences = []
    for sentence in sentences:
        tokens = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(tokens)):
            all_sequences.append(tokens[: i + 1])

    max_seq_len = max(len(s) for s in all_sequences)
    padded      = pad_sequences(all_sequences, maxlen=max_seq_len, padding="pre")

    X = padded[:, :-1]
    y = padded[:, -1]
    y = keras.utils.to_categorical(y, num_classes=vocab_size)

    print(f"  Training samples : {len(X)}")
    print(f"  Max seq length   : {max_seq_len}")
    print(f"  Input shape      : {X.shape}")

    # ── 4. Build model ────────────────────────────────────────────────
    print(f"\n[4/5] Building LSTM model …")
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, EMBED_DIM, input_length=max_seq_len - 1),
        keras.layers.LSTM(LSTM_UNITS, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(LSTM_UNITS),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(vocab_size, activation="softmax"),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        # Save best model based on val_accuracy
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Stop if no improvement for 10 epochs
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR if val_loss plateaus for 5 epochs
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── 5. Train ──────────────────────────────────────────────────────
    print(f"\n[5/5] Training — up to {EPOCHS} epochs …")
    print(f"  EarlyStopping patience : 10 epochs")
    print(f"  Validation split       : {int(VALIDATION_SPLIT*100)}%")
    print()

    start   = time.time()
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start

    # ── Save tokenizer ────────────────────────────────────────────────
    tokenizer._max_seq_len = max_seq_len
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    # ── Results ───────────────────────────────────────────────────────
    best_acc  = max(history.history.get("val_accuracy", [0]))
    best_loss = min(history.history.get("val_loss", [0]))
    epochs_run = len(history.history["loss"])

    print("\n" + "=" * 60)
    print("  Training Results")
    print("=" * 60)
    print(f"  Epochs run        : {epochs_run} / {EPOCHS}")
    print(f"  Best val accuracy : {best_acc*100:.1f}%")
    print(f"  Best val loss     : {best_loss:.4f}")
    print(f"  Training time     : {elapsed/60:.1f} minutes")
    print(f"  Model saved   →   {MODEL_PATH}")
    print(f"  Tokenizer saved → {TOKENIZER_PATH}")

    # Quality rating
    print()
    if best_acc >= 0.80:
        print("  Quality: EXCELLENT ✓ (above 80% val accuracy)")
    elif best_acc >= 0.60:
        print("  Quality: GOOD ✓ (above 60% — add more data to improve)")
    elif best_acc >= 0.40:
        print("  Quality: FAIR — add 50+ more sentences to corpus.py")
    else:
        print("  Quality: POOR — add 100+ more sentences to corpus.py")

    # ── Prediction test on all 8 fields ──────────────────────────────
    print("\n" + "=" * 60)
    print("  Prediction Test — All 8 Fields")
    print("=" * 60)

    test_seeds = {
        "Introduction":          "Hi I am a",
        "About Me / Bio":        "I am a passionate",
        "Skills":                "My skills include",
        "Projects":              "I built a",
        "Experience":            "I worked as a",
        "Additional Experience": "I volunteered as a",
        "Qualifications":        "I hold the",
        "Education Details":     "I hold a Bachelor",
    }

    for field, seed in test_seeds.items():
        prediction = test_predict(model, tokenizer, max_seq_len, seed)
        print(f"\n  [{field}]")
        print(f"  Seed      : \"{seed}\"")
        print(f"  Predicted : \"{seed} {prediction}\"")

    print("\n" + "=" * 60)
    print("  Done! Now run: python app.py")
    print("  The server will load the saved model instantly.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Always delete old model before retraining
    for path in [MODEL_PATH, TOKENIZER_PATH]:
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Deleted old file: {path}")

    train()