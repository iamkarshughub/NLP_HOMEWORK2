# CS5760 – Natural Language Processing
## Homework 2

**Student Name:** Shaik Karishma  
**Student ID:** 700768890




## Q5— Evaluation Metrics from Confusion Matrix

Goal

This task evaluates classification performance using precision and recall for multiple classes.

Input

The program uses a confusion matrix representing predicted vs true labels for three classes.

Each row corresponds to predictions and each column corresponds to actual labels.

Step 1 — Extract class statistics

For each class, the code computes:

True positives

False positives

False negatives

These values form the basis of evaluation metrics.

Step 2 — Compute precision and recall

Precision measures prediction correctness, while recall measures coverage of actual instances.

The program calculates these values separately for each class to understand class-specific performance.

Step 3 — Macro averaging

Macro averaging computes the mean of per-class metrics, giving equal importance to each class.

This is useful when class distributions are imbalanced.

Step 4 — Micro averaging

Micro averaging aggregates counts across all classes before computing metrics. This reflects overall system performance.

Output

The notebook prints:

precision and recall for each class

macro precision and recall

micro precision and recall

These outputs provide a complete evaluation of classifier behavior.


---

# Part II — Programming

## Q1) Bigram Language Model (MLE)
- Builds unigram + bigram counts from the 3 training sentences.
- Computes sentence probabilities for:
  - <s> I love NLP </s>
  - <s> I love deep learning </s>
- Prints which sentence is preferred (higher probability).

The objective of this task is to build a simple bigram language model that learns word transition probabilities from a training corpus and uses them to compute sentence probabilities.

Input

The program takes a small set of training sentences that include start and end tokens. These tokens allow the model to learn sentence boundaries.

Example training data contains sentences such as:

I love NLP

I love deep learning

deep learning is fun

Step 1 — Counting n-grams

The code first counts:

individual word occurrences (unigrams)

adjacent word pairs (bigrams)

This counting process builds the statistical foundation of the language model.

Step 2 — Computing bigram probabilities

The program uses Maximum Likelihood Estimation:

P(wᵢ | wᵢ₋₁) = C(wᵢ₋₁ , wᵢ) / C(wᵢ₋₁)

This means the probability of a word depends on how often it follows the previous word in the corpus.

Step 3 — Sentence probability calculation

To compute the probability of a sentence, the program multiplies the probabilities of each bigram sequence from start to end.

The program also prints each intermediate probability so the calculation can be traced.

This allows comparison between two sentences to determine which one the model considers more likely.

Output

The notebook shows:

unigram counts

bigram counts

probability of each test sentence

which sentence is preferred by the model

This demonstrates how language models evaluate fluency.


##How to Run

- Open notebook in Google Colab
- Run all cells
- Bigram language model: unigram/bigram counts, MLE probabilities, sentence probability for S1 and S2, and preference output.
- Confusion matrix metrics: per-class precision/recall, macro and micro averages, printed clearly.
