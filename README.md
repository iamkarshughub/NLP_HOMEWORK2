# CS5760 – Natural Language Processing
## Homework 2

**Student Name:** Shaik Karishma  
**Student ID:** 700768890



---

#Q1 — Naive Bayes Document Classification

The task is to classify a document using Naive Bayes with add-1 smoothing.

Formula Used

The Naive Bayes score for class 
𝑐
c:

Score(c) = P(c) × ∏ P(w | c)

where:

P(c) is the prior probability

P(w | c) is the likelihood of each word given the class

Add-1 smoothing:

P(w | c) = (count(w,c) + 1) / (N_c + |V|)

Example Classification

Sentence: very fun and predictable

Given:

P(−)=3/5, P(+)=2/5

Vocabulary size = 20

Negative tokens = 14 → denominator = 34

Positive tokens = 9 → denominator = 29

Likelihoods were computed for each word using smoothing and multiplied with priors.

After comparing scores, the negative class produced the larger value.

Final Result

The sentence is classified as Negative.

This demonstrates how Naive Bayes combines prior knowledge and word likelihoods.

#Q2) Harms of Classification

(a) Representational harm
Representational harm happens when a system reinforces stereotypes or unfair associations about a group through how it labels/represents them. In the Kiritchenko & Mohammad (2018) type of setup, this harm is shown when model outputs (e.g., emotion/association predictions) systematically differ across demographic identity terms, reflecting biased language patterns in training data and causing unfair representation.

(b) One risk of censorship in toxicity classification
A key risk is over-blocking or silencing: systems may label non-toxic content as toxic (especially when it contains identity terms or reclaimed language), causing communities to get unfairly moderated and reducing legitimate speech.

(c) Why worse on African American English / Indian English
Models often perform worse because training/test data is not balanced across dialects. If a model is mostly trained on “standard” English, dialect grammar/spelling/phrasing looks “unusual” to the model, increasing false positives/negatives.


#Q3) Bigram Probabilities + Zero Probability
(A) Sentence probabilities (MLE)
Bigram MLE: P(w|h) = C(h,w) / C(h)

From the table:
- P(I|<s>) = 2/3
- P(love|I) = 1
- P(NLP|love) = 1/2
- P(deep|love) = 1/2
- P(learning|deep) = 1
- P(</s>|NLP) = 1
- P(</s>|learning) = 1/2

**S1:** <s> I love NLP </s>
P(S1) = (2/3)·1·(1/2)·1 = 1/3

**S2:** <s> I love deep learning </s>
P(S2) = (2/3)·1·(1/2)·1·(1/2) = 1/6

**More probable:** S1.

(B) Zero-probability problem
MLE P(noodle|ate) = 0 because the bigram never appears.
This makes any sentence containing “ate noodle” have probability 0 (bad for sentence probability and perplexity).

**Add-1 smoothing** (given |V|=10, total after “ate” = 12):
P_add1(noodle|ate) = (0+1)/(12+10) = 1/22

#Q4 — Backoff Model

The purpose of this question is to compute language model probabilities when higher-order n-grams are missing. A backoff model first attempts to use trigram probability and, if the trigram is unseen, backs off to a bigram model.

Formula Used

Trigram Maximum Likelihood Estimation:

P(wᵢ | wᵢ₋₂ , wᵢ₋₁) = C(wᵢ₋₂ , wᵢ₋₁ , wᵢ) / C(wᵢ₋₂ , wᵢ₋₁)

If the trigram count is zero, the model backs off to the bigram probability:

P(wᵢ | wᵢ₋₁) = C(wᵢ₋₁ , wᵢ) / C(wᵢ₋₁)

(a) Compute P(cats | I, like)

From the corpus:

C(I, like, cats) = 1
C(I, like) = 2

Applying the trigram formula:

P(cats | I, like) = 1 / 2 = 0.5

Therefore, the trigram probability equals 0.5.

(b) Compute P(dogs | You, like) using trigram → bigram backoff

The trigram (You like dogs) does not appear in the corpus. Therefore, the trigram probability becomes zero and the model backs off to the bigram level.

From the corpus:

C(like, dogs) = 1
C(like) = 3

Applying the bigram formula:

P(dogs | like) = 1 / 3 ≈ 0.33

Hence, the backoff probability is approximately 0.33.
(c) Why backoff is necessary (important explanation)

In real text data, the number of possible trigrams is huge, so most valid trigrams won’t appear in a small training set.

Without backoff:

unseen trigram → probability becomes 0

then any sentence containing that trigram gets total probability 0

the model cannot compare sentences correctly (everything can collapse to zero)

With backoff:

the model still assigns a reasonable probability using bigrams or unigrams

results become more stable and realistic under sparse data conditions
	​


#Q5) Multi-class Metrics
Confusion matrix (System rows × Gold columns):

|        | Cat | Dog | Rabbit |
|--------|-----|-----|--------|
| Cat    | 5   | 10  | 5      |
| Dog    | 15  | 20  | 10     |
| Rabbit | 0   | 15  | 10     |

Row sums (pred): Cat=20, Dog=45, Rabbit=25
Col sums (gold): Cat=20, Dog=45, Rabbit=25
TP: Cat=5, Dog=20, Rabbit=10

Per-class:
- Cat: Precision=5/20=0.25, Recall=5/20=0.25
- Dog: Precision=20/45=0.4444, Recall=20/45=0.4444
- Rabbit: Precision=10/25=0.40, Recall=10/25=0.40

Macro Precision/Recall ≈ (0.25+0.4444+0.40)/3 ≈ 0.3648
Micro Precision/Recall = (5+20+10)/90 = 35/90 ≈ 0.3889

 Code prints all metrics clearly.

 Interpretation

Macro average is useful when class distribution is uneven, while micro average reflects overall system performance.


#Task 2 — Evaluation Metrics from Confusion Matrix
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
