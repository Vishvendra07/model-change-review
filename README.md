# Model Change Review

Train word and character TF-IDF classifiers, then inspect routing coverage, confusion matrices, and the predictions that changed.

[Source](project.py) · [Demo evidence](evidence/demo.json) · [Verification](evidence/checks.json)

## Run locally

Requires Python 3.11+.

```sh
python -m pip install -r requirements.txt
python project.py
python -m unittest discover -v
```

## Engineering decision

Evaluate on distinct text, report paired predictions and confidence thresholds, and make the cost of abstention visible.

A small synthetic text-routing experiment. The reported scores are model outputs on this dataset and confidence is uncalibrated.



[Technology or project reference](https://scikit-learn.org/stable/modules/feature_extraction.html)
