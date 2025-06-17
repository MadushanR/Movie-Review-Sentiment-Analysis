import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1) Load user feedback if present
try:
    user_data = pd.read_csv("user_feedback.csv", header=None, names=["review","sentiment"])
    user_data["sentiment"] = user_data["sentiment"].astype(int)
except FileNotFoundError:
    user_data = pd.DataFrame(columns=["review","sentiment"])

# 2) Load IMDb dataset and combine
df = pd.read_csv("IMDB Dataset.csv",on_bad_lines="skip",engine="python")
df["sentiment"] = df["sentiment"].map({"positive":1,"negative":0})
df = pd.concat([df, user_data], ignore_index=True).sample(frac=1, random_state=42)

# 3) Split into raw-text inputs and numeric labels
train_size   = int(0.8 * len(df))
train_texts  = df["review"][:train_size].values       # numpy array of strings
train_labels = df["sentiment"][:train_size].astype("int32").values
test_texts   = df["review"][train_size:].values
test_labels  = df["sentiment"][train_size:].astype("int32").values

# 4) Build and adapt TextVectorization
vocab_size = 10000
maxlen     = 300

vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=maxlen
)
vectorize_layer.adapt(train_texts)  # learns vocabulary and integer mapping

# 5) Build the Sequential model with vectorization as the first layer
model = Sequential([
    vectorize_layer,                      # raw string -> (batch, maxlen) int32
    Embedding(input_dim=vocab_size,
              output_dim=128),            # -> (batch, maxlen, 128)
    LSTM(64),                             # -> (batch, 64)
    Dense(1, activation="sigmoid")       # -> (batch, 1) probability
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 6) Callbacks: early stop + save best
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_lstm_textvec.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# 7) Train on raw strings directly
history = model.fit(
    train_texts, train_labels,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stopping, model_checkpoint]
)

# 8) Evaluate on raw-text test set
loss, acc = model.evaluate(test_texts, test_labels)
print(f"\nTest accuracy: {acc:.3f}")
