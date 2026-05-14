# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 2 | 4 | 0.50 |
| lexical_trigger | 2 | 7 | 0.29 |
| domain_shift | 0 | 7 | 0.00 |
| length_extreme | 3 | 6 | 0.50 |
| sarcasm | 0 | 4 | 0.00 |
| other | 0 | 2 | 0.00 |

Overall accuracy: 7 / 30 = 0.23

## Confirmed hypotheses

- **domain_shift**: All domain-shift examples (ids 3, 13, 14, 15, 16, 17, 18) were misclassified, mostly as positive. The model clearly struggles with out-of-domain sentences like sports scores, recipes, and weather forecasts.
- **sarcasm**: All sarcastic examples (ids 25, 26, 27, 28) were misclassified as positive, confirming that sarcasm remains a hard failure mode.
- **lexical_trigger**: Most examples (ids 7, 9, 10, 11, 12) were misclassified as positive because the model fixated on positive cue words (“amazing”, “beautiful”, “fantastic”) and ignored the negative context that follows.

## Refuted hypotheses

- **negation**: The model correctly classified ids 1 and 5 (both negation examples), which was better than expected. It seems to handle some negations like “did not improve” and “does not provide any useful features” correctly.
- **length_extreme**: The model correctly handled some very short examples (ids 19, 21) and a very long negative example (id 23), which refuted the hypothesis that all extreme-length examples would fail.

## What the results reveal about the decision boundary

The model’s decision boundary is highly reliant on surface-level positive lexical cues (“amazing”, “fantastic”, “great”) and often ignores negations or contextual reversals. It strongly favors labeling out-of-domain sentences as positive, and it completely fails to detect sarcasm. The model appears to have learned a very narrow set of app-review-specific patterns that don’t generalize well to other domains or nuanced language.
