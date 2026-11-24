"""Compare two trained offline ticket classifiers and measure abstention tradeoffs."""

import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline

TRAIN = {
    "billing": [
        "invoice amount is wrong",
        "need a refund for the charge",
        "unexpected payment today",
        "credit card was charged twice",
        "download the billing receipt",
        "request invoice for renewal",
        "please reverse the transaction",
        "failed payment on my card",
    ],
    "access": [
        "reset account password",
        "sign in screen is blank",
        "locked out of my workspace",
        "my account login expired",
        "email link to log in",
        "cannot access my dashboard",
        "forgot the account password",
        "enable two factor sign in",
    ],
    "schedule": [
        "reschedule my meeting",
        "change the appointment time",
        "calendar date is incorrect",
        "cancel my consultation",
        "need an earlier slot",
        "move our appointment tomorrow",
        "book a meeting next week",
        "update the calendar invite",
    ],
}
EVAL = [
    ("bill-01", "billing", "payment posted twice"),
    ("bill-02", "billing", "invoce renewal amount"),
    ("bill-03", "billing", "reverse this credit-card transaction"),
    ("bill-04", "billing", "I want the payment receipt"),
    ("bill-05", "billing", "charger refused my card"),
    ("bill-06", "billing", "account charged me after renewal"),
    ("access-01", "access", "sign-in link expired"),
    ("access-02", "access", "forgot my passwrd"),
    ("access-03", "access", "can’t log into dashboard"),
    ("access-04", "access", "workspace login locked"),
    ("access-05", "access", "reset account credentials"),
    ("access-06", "access", "my receipt opens but my account won’t"),
    ("schedule-01", "schedule", "appointment date is wrong"),
    ("schedule-02", "schedule", "rescedule tomorrow’s consultation"),
    ("schedule-03", "schedule", "move the meeting invite"),
    ("schedule-04", "schedule", "change to an earlier slot"),
    ("schedule-05", "schedule", "calendar invitation has wrong time"),
    ("schedule-06", "schedule", "please cancel the meeting and refund"),
    ("stress-01", "billing", "refnd reciept unexpcted"),
    ("stress-02", "access", "passwrd signn credentals"),
    ("stress-03", "schedule", "calendr appoitment rescedule"),
]
THRESHOLDS = (0.40, 0.50, 0.60, 0.70)


def evaluate(name, analyzer, ngrams):
    corpus = [(sample, label) for label, rows in TRAIN.items() for sample in rows]
    texts, labels = zip(*corpus, strict=True)
    model = make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams), LogisticRegression(random_state=17)
    )
    model.fit(texts, labels)
    sentences = [sentence for _, _, sentence in EVAL]
    start = time.perf_counter_ns()
    scores = model.predict_proba(sentences)
    elapsed = (time.perf_counter_ns() - start) / 1_000_000
    rows = []
    for (sample_id, actual, sentence), probabilities in zip(EVAL, scores, strict=True):
        index = int(probabilities.argmax())
        rows.append(
            {
                "id": sample_id,
                "text": sentence,
                "expected": actual,
                "predicted": str(model.classes_[index]),
                "score": float(probabilities[index]),
            }
        )
    actual = [row["expected"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    thresholds = {}
    for boundary in THRESHOLDS:
        selected = [row for row in rows if row["score"] >= boundary]
        thresholds[str(boundary)] = {
            "routed": len(selected),
            "coverage": round(len(selected) / len(rows), 4),
            "routed_accuracy": sum(row["expected"] == row["predicted"] for row in selected)
            / len(selected)
            if selected
            else None,
        }
    return {
        "name": name,
        "training_examples": len(corpus),
        "evaluation_examples": len(rows),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "classes": list(sorted(TRAIN)),
        "confusion": confusion_matrix(actual, predicted, labels=sorted(TRAIN)).tolist(),
        "batch_inference_ms": round(elapsed, 3),
        "thresholds": thresholds,
        "predictions": rows,
    }


def demo():
    baseline = evaluate("word TF-IDF", "word", (1, 2))
    candidate = evaluate("character TF-IDF", "char_wb", (3, 5))
    changes = [
        {
            "id": a["id"],
            "expected": a["expected"],
            "baseline": a["predicted"],
            "candidate": b["predicted"],
        }
        for a, b in zip(baseline["predictions"], candidate["predictions"], strict=True)
        if a["predicted"] != b["predicted"]
    ]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "changed_predictions": changes,
        "release_decision": "collect independent validation before routing customer tickets",
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
