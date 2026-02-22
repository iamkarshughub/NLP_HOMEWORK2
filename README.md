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

## Q3/Q4 (Additional)


Bigram Count Construction

The first step in the program is building a bigram frequency table. Each word is treated as a history term and the counts of words that follow it are stored. This allows the model to estimate the probability of the next word using observed frequencies.

For example, after the word “I”, the word “love” appears multiple times, which results in a higher probability for that transition.

This count table forms the basis for all later probability calculations.

Maximum Likelihood Estimation (MLE)

Using the count table, the program calculates bigram probabilities using the MLE formula:

P(w | h) = count(h,w) / total occurrences after h

This step measures how likely a word is to follow another word based purely on training data frequencies. The program computes these probabilities dynamically by summing counts for each history word.

Sentence Probability Computation

After computing bigram probabilities, the notebook evaluates complete sentences. The probability of a sentence is calculated by multiplying the probabilities of all bigram transitions within the sentence.

Two sentences are tested:

I love NLP

I love deep learning

The model computes likelihood for each sentence and selects the one with the higher probability as the preferred sentence.

This demonstrates how language models compare alternative sentences.

Zero Probability Problem

The notebook also checks a case where a bigram was never seen in training, such as noodle following ate. With MLE, this produces a probability of zero. This creates issues because one zero makes the entire sentence probability zero.

This is known as the data sparsity problem in N-gram models.

Add-One (Laplace) Smoothing

To address the zero probability issue, add-one smoothing is applied. This method adds one to every count and adjusts the denominator using the vocabulary size.

This ensures that unseen word pairs receive a small non-zero probability and the model can still evaluate sentences containing unseen combinations.

Backoff Concept

The notebook briefly demonstrates backoff behavior. When a bigram probability is unreliable or unseen, the model can rely on simpler probabilities such as unigram frequencies. This improves robustness when training data is limited.

Sample Outcome

The results show:

Sentence likelihood values for both test sentences

The preferred sentence selected by the model

MLE probability for an unseen bigram

Smoothed probability after applying Laplace smoothing

These outputs confirm the theoretical concepts discussed in class.


##How to Run

- Open notebook in Google Colab
- Run all cells
- Bigram language model: unigram/bigram counts, MLE probabilities, sentence probability for S1 and S2, and preference output.
- Confusion matrix metrics: per-class precision/recall, macro and micro averages, printed clearly.
