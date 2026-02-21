# CS5760 – Natural Language Processing
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

#Q2) Harms of Classification

(a) Representational harm
Representational harm happens when a system reinforces stereotypes or unfair associations about a group through how it labels/represents them. In the Kiritchenko & Mohammad (2018) type of setup, this harm is shown when model outputs (e.g., emotion/association predictions) systematically differ across demographic identity terms, reflecting biased language patterns in training data and causing unfair representation.

(b) One risk of censorship in toxicity classification
A key risk is over-blocking or silencing: systems may label non-toxic content as toxic (especially when it contains identity terms or reclaimed language), causing communities to get unfairly moderated and reducing legitimate speech.

(c) Why worse on African American English / Indian English
Models often perform worse because training/test data is not balanced across dialects. If a model is mostly trained on “standard” English, dialect grammar/spelling/phrasing looks “unusual” to the model, increasing false positives/negatives.


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

✅ **More probable:** S1.

### (B) Zero-probability problem
MLE P(noodle|ate) = 0 because the bigram never appears.
This makes any sentence containing “ate noodle” have probability 0 (bad for sentence probability and perplexity).

**Add-1 smoothing** (given |V|=10, total after “ate” = 12):
P_add1(noodle|ate) = (0+1)/(12+10) = 1/22


#Q4) Backoff Model

A backoff model means:

Try the highest-order model first (trigram).

If it’s unseen (probability 0), back off to a lower-order model (bigram).

If that’s also unseen, back off again (unigram).

✅ Key Formulas
1) Trigram MLE
𝑃
(
𝑤
𝑖
∣
𝑤
𝑖
−
2
,
𝑤
𝑖
−
1
)
=
𝐶
(
𝑤
𝑖
−
2
,
𝑤
𝑖
−
1
,
𝑤
𝑖
)
𝐶
(
𝑤
𝑖
−
2
,
𝑤
𝑖
−
1
)
P(w
i
	​

∣w
i−2
	​

,w
i−1
	​

)=
C(w
i−2
	​

,w
i−1
	​

)
C(w
i−2
	​

,w
i−1
	​

,w
i
	​

)
	​

2) Bigram MLE (backoff level 1)
𝑃
(
𝑤
𝑖
∣
𝑤
𝑖
−
1
)
=
𝐶
(
𝑤
𝑖
−
1
,
𝑤
𝑖
)
𝐶
(
𝑤
𝑖
−
1
)
P(w
i
	​

∣w
i−1
	​

)=
C(w
i−1
	​

)
C(w
i−1
	​

,w
i
	​

)
	​

3) Unigram MLE (backoff level 2)
𝑃
(
𝑤
𝑖
)
=
𝐶
(
𝑤
𝑖
)
𝑁
P(w
i
	​

)=
N
C(w
i
	​

)
	​

✅ (a) Compute 
𝑃
(
cats
∣
𝐼
,
𝑙
𝑖
𝑘
𝑒
)
P(cats∣I,like)

We first try trigram probability because we have a two-word history (I, like).

Given counts (from the question):

𝐶
(
𝐼
,
𝑙
𝑖
𝑘
𝑒
,
𝑐
𝑎
𝑡
𝑠
)
=
1
C(I,like,cats)=1

𝐶
(
𝐼
,
𝑙
𝑖
𝑘
𝑒
)
=
2
C(I,like)=2

Apply trigram formula:

𝑃
(
𝑐
𝑎
𝑡
𝑠
∣
𝐼
,
𝑙
𝑖
𝑘
𝑒
)
=
𝐶
(
𝐼
,
𝑙
𝑖
𝑘
𝑒
,
𝑐
𝑎
𝑡
𝑠
)
𝐶
(
𝐼
,
𝑙
𝑖
𝑘
𝑒
)
=
1
2
=
0.5
P(cats∣I,like)=
C(I,like)
C(I,like,cats)
	​

=
2
1
	​

=0.5

✅ Answer: 
0.5
0.5
	​


✅ (b) Compute 
𝑃
(
dogs
∣
𝑌
𝑜
𝑢
,
𝑙
𝑖
𝑘
𝑒
)
P(dogs∣You,like) using trigram → bigram backoff
Step 1 — Try trigram first

We check:

𝐶
(
𝑌
𝑜
𝑢
,
𝑙
𝑖
𝑘
𝑒
,
𝑑
𝑜
𝑔
𝑠
)
C(You,like,dogs)

The trigram (You like dogs) does not appear in the corpus, so:

𝐶
(
𝑌
𝑜
𝑢
,
𝑙
𝑖
𝑘
𝑒
,
𝑑
𝑜
𝑔
𝑠
)
=
0
⇒
𝑃
(
𝑑
𝑜
𝑔
𝑠
∣
𝑌
𝑜
𝑢
,
𝑙
𝑖
𝑘
𝑒
)
=
0
C(You,like,dogs)=0⇒P(dogs∣You,like)=0

That means trigram MLE fails (zero probability), so we back off.

Step 2 — Back off to bigram

Now we compute using only the most recent word like:

Given counts:

𝐶
(
𝑙
𝑖
𝑘
𝑒
,
𝑑
𝑜
𝑔
𝑠
)
=
1
C(like,dogs)=1

𝐶
(
𝑙
𝑖
𝑘
𝑒
)
=
3
C(like)=3

Apply bigram formula:

𝑃
(
𝑑
𝑜
𝑔
𝑠
∣
𝑙
𝑖
𝑘
𝑒
)
=
𝐶
(
𝑙
𝑖
𝑘
𝑒
,
𝑑
𝑜
𝑔
𝑠
)
𝐶
(
𝑙
𝑖
𝑘
𝑒
)
=
1
3
≈
0.333
P(dogs∣like)=
C(like)
C(like,dogs)
	​

=
3
1
	​

≈0.333

✅ Answer (with backoff):

𝑃
(
𝑑
𝑜
𝑔
𝑠
∣
𝑌
𝑜
𝑢
,
𝑙
𝑖
𝑘
𝑒
)
≈
1
3
P(dogs∣You,like)≈
3
1
	​

	​

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
