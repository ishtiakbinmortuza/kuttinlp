# ============================================================
# Kutti Dialect -> Standard Bangla | Data loading, cleaning, augmentation
# ============================================================

import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# ---------------- CONFIG ----------------
KUTTI_SENT_PATH = "Kutti_Dataset_Sentences.csv"
KUTTI_DICT_PATH = "Kutti_Dataset_Dictionary.csv"

RANDOM_SEED = 42
VAL_SIZE = 0.1
TEST_SIZE = 0.1
DICT_UPSAMPLE = 6
LEXICON_INJECT_COPIES = 2

MODEL_NAME = "csebuetnlp/banglat5"
OUTPUT_DIR = "banglat5-kutti"
MAX_SOURCE_LEN = 64
MAX_TARGET_LEN = 64

NUM_EPOCHS = 15
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EARLY_STOPPING_PATIENCE = 5


# ============================================================
# 1) PREPROCESSING
# ============================================================

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u200b", "")
    return text

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["kutti", "standard"]
    df["kutti"] = df["kutti"].apply(normalize_text)
    df["standard"] = df["standard"].apply(normalize_text)
    df = df[(df["kutti"].str.len() > 0) & (df["standard"].str.len() > 0)]
    same_mask = df["kutti"].str.strip().str.casefold() == df["standard"].str.strip().str.casefold()
    n_dropped = int(same_mask.sum())
    if n_dropped:
        print(f"Dropping {n_dropped} row(s) where kutti == standard (no signal to learn)")
    df = df[~same_mask]
    # Drop exact duplicates BEFORE splitting so the same pair can't land in both train and test
    n_dupes = int(df.duplicated(subset=["kutti", "standard"]).sum())
    if n_dupes:
        print(f"Dropping {n_dupes} duplicate row(s)")
    df = df.drop_duplicates(subset=["kutti", "standard"])
    df = df[df["kutti"].str.len() <= 200]
    return df.reset_index(drop=True)

print("Loading & cleaning data...")
sent_df = clean_dataframe(pd.read_csv(KUTTI_SENT_PATH))
dict_df = clean_dataframe(pd.read_csv(KUTTI_DICT_PATH))
print(f"Sentence pairs: {len(sent_df)} | Word-dictionary pairs: {len(dict_df)}")

# Split sentence data FIRST (dictionary rows only ever go into train -> no leakage risk there either)
train_sent, temp_sent = train_test_split(
    sent_df, test_size=(VAL_SIZE + TEST_SIZE), random_state=RANDOM_SEED
)
val_sent, test_sent = train_test_split(
    temp_sent, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), random_state=RANDOM_SEED
)

# ---- Lexicon-injection augmentation ----
# For each dictionary pair, mine TRAIN sentences containing the dialect word as a whole token
# and add a variant where just that word is substituted. This teaches the model the word-level
# mapping *in context*, which generalizes better than isolated word pairs alone.
def tokenize(text):
    return re.findall(r"\S+", text)

def inject_word_pairs(base_df, dict_df, max_copies):
    augmented = []
    base_tokenized = [(row.kutti, row.standard, tokenize(row.kutti)) for row in base_df.itertuples()]
    for d in dict_df.itertuples():
        src_word, tgt_word = d.kutti, d.standard
        count = 0
        for kutti_sent, std_sent, toks in base_tokenized:
            if count >= max_copies:
                break
            if src_word in toks and src_word not in std_sent:
                new_kutti = " ".join(tgt_word if t == src_word else t for t in tokenize(kutti_sent))
                # only keep if it actually changed something new (avoid pure duplicates)
                if new_kutti != kutti_sent:
                    augmented.append({"kutti": new_kutti, "standard": std_sent})
                    count += 1
    return pd.DataFrame(augmented)

lexicon_aug_df = inject_word_pairs(train_sent, dict_df, LEXICON_INJECT_COPIES)
lexicon_aug_df = clean_dataframe(lexicon_aug_df) if len(lexicon_aug_df) else lexicon_aug_df
print(f"Lexicon-injection augmented pairs: {len(lexicon_aug_df)}")

dict_upsampled = pd.concat([dict_df] * DICT_UPSAMPLE, ignore_index=True)
train_df = pd.concat([train_sent, dict_upsampled, lexicon_aug_df], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

def to_source_target(df):
    out = df[["kutti", "standard"]].copy()
    out.columns = ["source", "target"]
    return out

train_out = to_source_target(train_df)
val_out = to_source_target(val_sent)
test_out = to_source_target(test_sent)

print(f"train: {len(train_out)} | val: {len(val_out)} | test: {len(test_out)}")

train_ds = Dataset.from_pandas(train_out, preserve_index=False)
val_ds = Dataset.from_pandas(val_out, preserve_index=False)
test_ds = Dataset.from_pandas(test_out, preserve_index=False)

# Build a lookup dictionary for the lexicon post-edit step used later
LEXICON = dict(zip(dict_df["kutti"], dict_df["standard"]))

# ============================================================
# 2) FINE-TUNE BanglaT5 (pretrained seq2seq, not a randomly-initialized cross-attention model)
# ============================================================
import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

USE_FP16 = torch.cuda.is_available()
if not torch.cuda.is_available():
    print("WARNING: no GPU detected. Training will be very slow on CPU — use a GPU runtime.")

print("Loading tokenizer & BanglaT5 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess_fn(batch):
    model_inputs = tokenizer(
        batch["source"], max_length=MAX_SOURCE_LEN, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        text_target=batch["target"], max_length=MAX_TARGET_LEN, truncation=True, padding="max_length"
    )
    label_ids = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs

train_ds_tok = train_ds.map(preprocess_fn, batched=True, remove_columns=["source", "target"])
val_ds_tok = val_ds.map(preprocess_fn, batched=True, remove_columns=["source", "target"])
test_ds_tok = test_ds.map(preprocess_fn, batched=True, remove_columns=["source", "target"])

cer_metric = evaluate.load("cer")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    exact_match = np.mean([p.strip() == l.strip() for p, l in zip(pred_str, label_str)])
    return {"cer": cer, "exact_match": exact_match}

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    generation_num_beams=4,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    greater_is_better=True,
    fp16=False,
    label_smoothing_factor=0.1,
    report_to="none",
)

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_ds_tok,
    eval_dataset=val_ds_tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)
try:
    trainer = Seq2SeqTrainer(processing_class=tokenizer, **trainer_kwargs)
except TypeError:
    trainer = Seq2SeqTrainer(tokenizer=tokenizer, **trainer_kwargs)

print("Training...")
trainer.train()

trainer.save_model(OUTPUT_DIR + "/best")
tokenizer.save_pretrained(OUTPUT_DIR + "/best")
print(f"Model saved to {OUTPUT_DIR}/best")

test_metrics_model_only = trainer.evaluate(eval_dataset=test_ds_tok, metric_key_prefix="test")
print("Test metrics (model only):", test_metrics_model_only)

# ============================================================
# 3) INFERENCE + LEXICON POST-EDIT (reported separately as an ablation, not hidden)
# ============================================================
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def translate(text: str) -> str:
    text = normalize_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_LEN).to(device)
    output_ids = model.generate(**inputs, max_length=MAX_TARGET_LEN, num_beams=4, no_repeat_ngram_size=3)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def lexicon_post_edit(text: str) -> str:
    """Swap any remaining known dialect words for their standard form.
    This is a deterministic, dictionary-driven correction pass -- it can only ever
    replace a word with the exact standard form YOU supplied in the dictionary,
    so it cannot leak test-set answers; it just catches words the model missed."""
    toks = tokenize(text)
    fixed = [LEXICON.get(tok, tok) for tok in toks]
    return " ".join(fixed)

def translate_hybrid(text: str) -> str:
    return lexicon_post_edit(translate(text))

print("\n--- Sample translations ---")
for ex in ["আমার ফুন ভাই", "আপনে আমারে নগদ দিবেন", "অহন আমি ফুনের"]:
    print(f"KUTTI      : {ex}")
    print(f"MODEL      : {translate(ex)}")
    print(f"MODEL+LEX  : {translate_hybrid(ex)}\n")

# ============================================================
# 4) FULL EVALUATION REPORT (dataset stats, training curves, model-only vs hybrid metrics)
# ============================================================
import matplotlib.pyplot as plt
from IPython.display import display
from collections import defaultdict
from sacrebleu.metrics import BLEU, CHRF, TER

# 1) Dataset split summary
split_summary = pd.DataFrame({
    "split": ["train", "val", "test"],
    "rows": [len(train_out), len(val_out), len(test_out)],
})
print("Dataset split summary")
display(split_summary)

plt.figure(figsize=(6, 3))
plt.bar(split_summary["split"], split_summary["rows"], color=["#4C78A8", "#F58518", "#54A24B"])
plt.title("Dataset split sizes")
plt.ylabel("Number of examples")
plt.tight_layout()
plt.show()

# 2) Training/evaluation history
epoch_metrics = defaultdict(dict)
for entry in trainer.state.log_history:
    epoch = entry.get("epoch")
    if epoch is None:
        continue
    for key in ("loss", "eval_loss", "eval_cer", "eval_exact_match"):
        if key in entry:
            epoch_metrics[epoch][key] = entry[key]

history_df = pd.DataFrame.from_dict(epoch_metrics, orient="index").reset_index().rename(columns={"index": "epoch"})
history_df = history_df.sort_values("epoch").reset_index(drop=True)
print("\nTraining and evaluation history")
display(history_df)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_df["epoch"], history_df["loss"], marker="o", label="train loss")
plt.plot(history_df["epoch"], history_df["eval_loss"], marker="o", label="eval loss")
plt.title("Loss across epochs")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_df["epoch"], history_df["eval_cer"], marker="o", label="CER")
plt.plot(history_df["epoch"], history_df["eval_exact_match"], marker="o", label="Exact match")
plt.title("Validation metrics across epochs")
plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.legend()
plt.tight_layout()
plt.show()

print("\nBest checkpoint:", trainer.state.best_model_checkpoint)
print("Best validation metric:", trainer.state.best_metric)

# 3) Build predictions on the test set -- model-only AND hybrid (model + lexicon post-edit)
preds_model = [translate(s) for s in test_out["source"]]
preds_hybrid = [lexicon_post_edit(p) for p in preds_model]
refs = list(test_out["target"])

def score_set(preds, refs, label):
    exact_match_accuracy = float(np.mean([p.strip() == r.strip() for p, r in zip(preds, refs)]))
    
    all_tokens = sorted({tok for p, r in zip(preds, refs) for tok in tokenize(p) + tokenize(r)})
    rows = []
    for tok in all_tokens:
        tp = sum(1 for p, r in zip(preds, refs) if tok in tokenize(p) and tok in tokenize(r))
        fp = sum(1 for p, r in zip(preds, refs) if tok in tokenize(p) and tok not in tokenize(r))
        fn = sum(1 for p, r in zip(preds, refs) if tok not in tokenize(p) and tok in tokenize(r))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"token": tok, "precision": precision, "recall": recall, "f1": f1, "support": tp + fn})
    token_report = pd.DataFrame(rows)
    macro_summary = pd.DataFrame([{
        "precision": token_report["precision"].mean(),
        "recall": token_report["recall"].mean(),
        "f1": token_report["f1"].mean(),
        "support": int(token_report["support"].sum()),
    }], index=["macro_avg"])
    
    bleu = BLEU(effective_order=True); chrf = CHRF(); ter = TER()
    bleu_score = bleu.corpus_score(preds, [refs]).score
    chrf_score = chrf.corpus_score(preds, [refs]).score
    ter_score = ter.corpus_score(preds, [refs]).score

    print(f"\n=== Evaluation Report: {label} ===")
    print(f"Exact-match accuracy: {exact_match_accuracy:.4f}")
    print(f"Macro precision: {macro_summary.loc['macro_avg', 'precision']:.4f}")
    print(f"Macro recall: {macro_summary.loc['macro_avg', 'recall']:.4f}")
    print(f"Macro F1: {macro_summary.loc['macro_avg', 'f1']:.4f}")
    print(f"SacreBLEU: {bleu_score:.4f}")
    print(f"ChrF: {chrf_score:.4f}")
    print(f"TER: {ter_score:.4f}")
    return exact_match_accuracy, token_report

acc_model, token_report_model = score_set(preds_model, refs, "Model only (BanglaT5 fine-tuned)")
acc_hybrid, token_report_hybrid = score_set(preds_hybrid, refs, "Model + lexicon post-edit (hybrid)")

# 4) Sample predictions on the held-out test set
sample_rows = test_out.sample(min(10, len(test_out)), random_state=42).copy()
sample_rows["model_pred"] = sample_rows["source"].apply(translate)
sample_rows["hybrid_pred"] = sample_rows["model_pred"].apply(lexicon_post_edit)
print("\nSample predictions on the test set")
display(sample_rows[["source", "target", "model_pred", "hybrid_pred"]])
