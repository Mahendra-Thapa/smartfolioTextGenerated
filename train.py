"""
train.py
--------
Deep learning optimized LSTM training script for SmartFolioAI.

✔ CONTINUE TRAINING support
✔ Temperature-based predictions to avoid repetitions
✔ Runs all epochs without early stopping
"""

import os
import sys
import pickle
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import CORPUS
from simple_tokenizer import SimpleTokenizer

# ── Config ─────────────────────────────────────────
EPOCHS           = 60
BATCH_SIZE       = 16
EMBED_DIM        = 128
LSTM_UNITS       = 256
VALIDATION_SPLIT = 0.1
TEMPERATURE      = 1.0  # Higher value = more diversity

MODEL_DIR      = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_PATH     = os.path.join(MODEL_DIR, "lstm.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# ── Padding ────────────────────────────────────────
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

# ── Sampling with Temperature ───────────────────────
def sample_with_temperature(preds, temperature=TEMPERATURE):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# ── Prediction Test ────────────────────────────────
def test_predict(model, tokenizer, max_seq_len, seed, n_words=10):
    text = seed.lower()
    result = []
    for _ in range(n_words):
        tokens = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([tokens], maxlen=max_seq_len - 1)
        probs = model.predict(padded, verbose=0)[0]
        next_index = sample_with_temperature(probs)
        word = tokenizer.index_word.get(next_index, "")
        if not word:
            break
        result.append(word)
        text += " " + word
    return " ".join(result)

# ── Main Training ──────────────────────────────────
def train():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras import callbacks, Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

    keras = tf.keras

    # Enable GPU memory growth if GPU exists
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("\n" + "=" * 60)
    print(" SmartFolioAI — LSTM Training (FULL EPOCHS)")
    print("=" * 60)

    # ── 1. Load Data ───────────────────────────────
    sentences = CORPUS[:]
    print(f"\n[1/5] Data Summary")
    print(f"  Total sentences : {len(sentences)}")

    # ── 2. Tokenizer ──────────────────────────────
    print(f"\n[2/5] Tokenizer")
    if os.path.exists(TOKENIZER_PATH):
        print("  ✔ Loading existing tokenizer...")
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        print("  ✔ Creating new tokenizer...")
        tokenizer = SimpleTokenizer()
        tokenizer.fit(sentences)

    vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size : {vocab_size}")

    # ── 3. Sequences ──────────────────────────────
    print(f"\n[3/5] Building sequences...")
    all_sequences = []

    for sentence in sentences:
        tokens = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(tokens)):
            all_sequences.append(tokens[: i + 1])

    max_seq_len = max(len(s) for s in all_sequences)

    padded = pad_sequences(all_sequences, maxlen=max_seq_len)
    X = padded[:, :-1]
    y = padded[:, -1]

    y = keras.utils.to_categorical(y, num_classes=vocab_size)
    print(f"  Samples : {len(X)}")

    # ── 4. Model ──────────────────────────────────
    print(f"\n[4/5] Model")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("  ✔ Loading existing model (continue training)...")
        model = load_model(MODEL_PATH)
    else:
        print("  ✔ Creating new model...")
        model = Sequential([
            Embedding(vocab_size, EMBED_DIM),
            LSTM(LSTM_UNITS, return_sequences=True),
            Dropout(0.3),
            LSTM(LSTM_UNITS),
            Dropout(0.3),
            Dense(vocab_size, activation="softmax"),
        ])
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )

    model.summary()

    # ── Callbacks ────────────────────────────────
    cb = [
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # EarlyStopping removed to allow full 60 epochs
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ── 5. Train ─────────────────────────────────
    print(f"\n[5/5] Training...")

    start = time.time()
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=cb,
        verbose=1,
        shuffle=True
    )

    print(f"\nTraining time: {(time.time() - start)/60:.2f} minutes")
    print(f"Epochs completed: {len(history.history['loss'])}")

    # ── Save tokenizer ───────────────────────────
    tokenizer._max_seq_len = max_seq_len
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"\n✔ Model saved → {MODEL_PATH}")
    print(f"✔ Tokenizer saved → {TOKENIZER_PATH}")

    # ── Test Predictions ─────────────────────────
    print("\nSample Predictions:\n")
    seeds = [
    "Early exposure to computing as an academic foundation", "Early exposure to computing through guided instruction", "Early exposure to computing supported by practical exercises", "Early exposure to computing within structured learning environments", "Early exposure to computing across diverse coursework", "Early exposure to computing shaped by hands-on experience", "Early exposure to computing reinforced by iterative practice", "Early exposure to computing focused on clarity and structure", "Early exposure to computing aimed at practical understanding", "Early exposure to computing aligned with foundational principles", "Foundational study in technology as an academic foundation", "Foundational study in technology through guided instruction", "Foundational study in technology supported by practical exercises", "Foundational study in technology within structured learning environments", "Foundational study in technology across diverse coursework", "Foundational study in technology shaped by hands-on experience", "Foundational study in technology reinforced by iterative practice", "Foundational study in technology focused on clarity and structure", "Foundational study in technology aimed at practical understanding", "Foundational study in technology aligned with foundational principles", "Hands-on practice with software as an academic foundation", "Hands-on practice with software through guided instruction", "Hands-on practice with software supported by practical exercises", "Hands-on practice with software within structured learning environments", "Hands-on practice with software across diverse coursework", "Hands-on practice with software shaped by hands-on experience", "Hands-on practice with software reinforced by iterative practice", "Hands-on practice with software focused on clarity and structure", "Hands-on practice with software aimed at practical understanding", "Hands-on practice with software aligned with foundational principles", "Academic engagement in computer science as an academic foundation", "Academic engagement in computer science through guided instruction", "Academic engagement in computer science supported by practical exercises", "Academic engagement in computer science within structured learning environments", "Academic engagement in computer science across diverse coursework", "Academic engagement in computer science shaped by hands-on experience", "Academic engagement in computer science reinforced by iterative practice", "Academic engagement in computer science focused on clarity and structure", "Academic engagement in computer science aimed at practical understanding", "Academic engagement in computer science aligned with foundational principles", "Curiosity about digital systems as an academic foundation", "Curiosity about digital systems through guided instruction", "Curiosity about digital systems supported by practical exercises", "Curiosity about digital systems within structured learning environments", "Curiosity about digital systems across diverse coursework", "Curiosity about digital systems shaped by hands-on experience", "Curiosity about digital systems reinforced by iterative practice", "Curiosity about digital systems focused on clarity and structure", "Curiosity about digital systems aimed at practical understanding", "Curiosity about digital systems aligned with foundational principles", "Structured learning in programming as an academic foundation", "Structured learning in programming through guided instruction", "Structured learning in programming supported by practical exercises", "Structured learning in programming within structured learning environments", "Structured learning in programming across diverse coursework", "Structured learning in programming shaped by hands-on experience", "Structured learning in programming reinforced by iterative practice", "Structured learning in programming focused on clarity and structure", "Structured learning in programming aimed at practical understanding", "Structured learning in programming aligned with foundational principles", "Problem-solving driven coursework as an academic foundation", "Problem-solving driven coursework through guided instruction", "Problem-solving driven coursework supported by practical exercises", "Problem-solving driven coursework within structured learning environments", "Problem-solving driven coursework across diverse coursework", "Problem-solving driven coursework shaped by hands-on experience", "Problem-solving driven coursework reinforced by iterative practice", "Problem-solving driven coursework focused on clarity and structure", "Problem-solving driven coursework aimed at practical understanding", "Problem-solving driven coursework aligned with foundational principles", "Exploration of application behavior as an academic foundation", "Exploration of application behavior through guided instruction", "Exploration of application behavior supported by practical exercises", "Exploration of application behavior within structured learning environments", "Exploration of application behavior across diverse coursework", "Exploration of application behavior shaped by hands-on experience", "Exploration of application behavior reinforced by iterative practice", "Exploration of application behavior focused on clarity and structure", "Exploration of application behavior aimed at practical understanding", "Exploration of application behavior aligned with foundational principles", "Systematic study of algorithms as an academic foundation", "Systematic study of algorithms through guided instruction", "Systematic study of algorithms supported by practical exercises", "Systematic study of algorithms within structured learning environments", "Systematic study of algorithms across diverse coursework", "Systematic study of algorithms shaped by hands-on experience", "Systematic study of algorithms reinforced by iterative practice", "Systematic study of algorithms focused on clarity and structure", "Systematic study of algorithms aimed at practical understanding", "Systematic study of algorithms aligned with foundational principles", "Technical training across disciplines as an academic foundation", "Technical training across disciplines through guided instruction", "Technical training across disciplines supported by practical exercises", "Technical training across disciplines within structured learning environments", "Technical training across disciplines across diverse coursework", "Technical training across disciplines shaped by hands-on experience", "Technical training across disciplines reinforced by iterative practice", "Technical training across disciplines focused on clarity and structure", "Technical training across disciplines aimed at practical understanding", "Technical training across disciplines aligned with foundational principles", "Practical assignments in development as an academic foundation", "Practical assignments in development through guided instruction", "Practical assignments in development supported by practical exercises", "Practical assignments in development within structured learning environments", "Practical assignments in development across diverse coursework", "Practical assignments in development shaped by hands-on experience", "Practical assignments in development reinforced by iterative practice", "Practical assignments in development focused on clarity and structure", "Practical assignments in development aimed at practical understanding", "Practical assignments in development aligned with foundational principles", "Methodical approach to coding as an academic foundation", "Methodical approach to coding through guided instruction", "Methodical approach to coding supported by practical exercises", "Methodical approach to coding within structured learning environments", "Methodical approach to coding across diverse coursework", "Methodical approach to coding shaped by hands-on experience", "Methodical approach to coding reinforced by iterative practice", "Methodical approach to coding focused on clarity and structure", "Methodical approach to coding aimed at practical understanding", "Methodical approach to coding aligned with foundational principles", "Analytical thinking through software as an academic foundation", "Analytical thinking through software through guided instruction", "Analytical thinking through software supported by practical exercises", "Analytical thinking through software within structured learning environments", "Analytical thinking through software across diverse coursework", "Analytical thinking through software shaped by hands-on experience", "Analytical thinking through software reinforced by iterative practice", "Analytical thinking through software focused on clarity and structure", "Analytical thinking through software aimed at practical understanding", "Analytical thinking through software aligned with foundational principles", "Logical reasoning in applications as an academic foundation", "Logical reasoning in applications through guided instruction", "Logical reasoning in applications supported by practical exercises", "Logical reasoning in applications within structured learning environments", "Logical reasoning in applications across diverse coursework", "Logical reasoning in applications shaped by hands-on experience", "Logical reasoning in applications reinforced by iterative practice", "Logical reasoning in applications focused on clarity and structure", "Logical reasoning in applications aimed at practical understanding", "Logical reasoning in applications aligned with foundational principles", "Experimentation with data and logic as an academic foundation", "Experimentation with data and logic through guided instruction", "Experimentation with data and logic supported by practical exercises", "Experimentation with data and logic within structured learning environments", "Experimentation with data and logic across diverse coursework", "Experimentation with data and logic shaped by hands-on experience", "Experimentation with data and logic reinforced by iterative practice", "Experimentation with data and logic focused on clarity and structure", "Experimentation with data and logic aimed at practical understanding", "Experimentation with data and logic aligned with foundational principles", "Exposure to modern development tools as an academic foundation", "Exposure to modern development tools through guided instruction", "Exposure to modern development tools supported by practical exercises", "Exposure to modern development tools within structured learning environments", "Exposure to modern development tools across diverse coursework", "Exposure to modern development tools shaped by hands-on experience", "Exposure to modern development tools reinforced by iterative practice", "Exposure to modern development tools focused on clarity and structure", "Exposure to modern development tools aimed at practical understanding", "Exposure to modern development tools aligned with foundational principles", "Consistent practice in code design as an academic foundation", "Consistent practice in code design through guided instruction", "Consistent practice in code design supported by practical exercises", "Consistent practice in code design within structured learning environments", "Consistent practice in code design across diverse coursework", "Consistent practice in code design shaped by hands-on experience", "Consistent practice in code design reinforced by iterative practice", "Consistent practice in code design focused on clarity and structure", "Consistent practice in code design aimed at practical understanding", "Consistent practice in code design aligned with foundational principles", "Conceptual learning through projects as an academic foundation", "Conceptual learning through projects through guided instruction", "Conceptual learning through projects supported by practical exercises", "Conceptual learning through projects within structured learning environments", "Conceptual learning through projects across diverse coursework", "Conceptual learning through projects shaped by hands-on experience", "Conceptual learning through projects reinforced by iterative practice", "Conceptual learning through projects focused on clarity and structure", "Conceptual learning through projects aimed at practical understanding", "Conceptual learning through projects aligned with foundational principles", "Iterative refinement of solutions as an academic foundation", "Iterative refinement of solutions through guided instruction", "Iterative refinement of solutions supported by practical exercises", "Iterative refinement of solutions within structured learning environments", "Iterative refinement of solutions across diverse coursework", "Iterative refinement of solutions shaped by hands-on experience", "Iterative refinement of solutions reinforced by iterative practice", "Iterative refinement of solutions focused on clarity and structure", "Iterative refinement of solutions aimed at practical understanding", "Iterative refinement of solutions aligned with foundational principles", "Applied study of computing principles as an academic foundation", "Applied study of computing principles through guided instruction", "Applied study of computing principles supported by practical exercises", "Applied study of computing principles within structured learning environments", "Applied study of computing principles across diverse coursework", "Applied study of computing principles shaped by hands-on experience", "Applied study of computing principles reinforced by iterative practice", "Applied study of computing principles focused on clarity and structure", "Applied study of computing principles aimed at practical understanding", "Applied study of computing principles aligned with foundational principles",
]

    for seed in seeds:
        result = test_predict(model, tokenizer, max_seq_len, seed)
        print(f"{seed} → {seed} {result}")

    print("\n✅ Done! Run: python app.py\n")

# ── Entry Point ───────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print("\n[INFO] Existing model found → continuing training\n")
    else:
        print("\n[INFO] No model found → training from scratch\n")

    train()