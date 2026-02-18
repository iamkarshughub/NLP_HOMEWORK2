# CS5760 – Natural Language Processing (Spring 2026)
## Homework 2

**Student Name:** Shaik Karishma  
**Student ID:** 700768890



---

# Part I — Writing / Calculations 

## Q1) Worked Example: Classify “predictable no fun”
Naive Bayes score:

Score(c) = P(c) × Π P(w_i | c)

For **“predictable no fun”**:
- Score(pos) = P(pos)·P(predictable|pos)·P(no|pos)·P(fun|pos)
- Score(neg) = P(neg)·P(predictable|neg)·P(no|neg)·P(fun|neg)

 Pick the class with the larger score. (Notebook computes both scores once you plug in the given likelihoods from the slide/Q2.)

## Q2) Harms of Classification 
- **Representational harm:** systems reinforce stereotypes/biased associations about groups (Kiritchenko & Mohammad, 2018).
- **Censorship risk:** toxicity filters may over-flag identity-related or reclaimed terms, silencing legitimate speech (Dixon et al., 2018; Oliva et al., 2021).
- **Why worse on AAE/Indian English:** training data mismatch + dialect features (domain shift) → more false positives/negatives.

## Q3) Bigram Probabilities + Zero Probability
### (A) Sentence probabilities (MLE)
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

### (B) Zero-probability problem
MLE P(noodle|ate) = 0 because the bigram never appears.
This makes any sentence containing “ate noodle” have probability 0 (bad for sentence probability and perplexity).

**Add-1 smoothing** (given |V|=10, total after “ate” = 12):
P_add1(noodle|ate) = (0+1)/(12+10) = 1/22

## Q4) Backoff Model
Counts: I like = 2, You like = 1, like cats = 2, like dogs = 1

1) P(cats|I,like) = C(I like cats) / C(I like) = 1/2
2) P(dogs|You,like): trigram unseen → backoff to bigram
   P(dogs|like) = C(like dogs) / C(like) = 1/(2+1) = 1/3
3) Backoff is needed because small corpora have many unseen trigrams; backoff avoids zero probabilities.

## Q5) Multi-class Metrics
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

---

# Part II — Programming

## Q1) Bigram Language Model (MLE)
- Builds unigram + bigram counts from the 3 training sentences.
- Computes sentence probabilities for:
  - <s> I love NLP </s>
  - <s> I love deep learning </s>
- Prints which sentence is preferred (higher probability).

## How to Run
1. Open the notebook: **CS5760_HW2_Shaik_Karishma_700768890.ipynb**
2. Run all cells top-to-bottom.
3. Outputs will print in the notebook.
