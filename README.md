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

Q3) Bigram Probabilities and Zero-Probability Problem 


A) Sentence probability (MLE)

MLE bigram formula:

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


S1: 
⟨
𝑠
⟩
 
𝐼
 
𝑙
𝑜
𝑣
𝑒
 
𝑁
𝐿
𝑃
 
⟨
/
𝑠
⟩
⟨s⟩ I love NLP ⟨/s⟩

𝑃
(
𝐼
∣
⟨
𝑠
⟩
)
=
2
/
3
P(I∣⟨s⟩)=2/3

𝑃
(
𝑙
𝑜
𝑣
𝑒
∣
𝐼
)
=
2
/
2
=
1
P(love∣I)=2/2=1

𝑃
(
𝑁
𝐿
𝑃
∣
𝑙
𝑜
𝑣
𝑒
)
=
1
/
2
P(NLP∣love)=1/2

𝑃
(
⟨
/
𝑠
⟩
∣
𝑁
𝐿
𝑃
)
=
1
/
1
=
1
P(⟨/s⟩∣NLP)=1/1=1

𝑃
(
𝑆
1
)
=
2
3
⋅
1
⋅
1
2
⋅
1
=
1
3
P(S1)=
3
2
	​

⋅1⋅
2
1
	​

⋅1=
3
1
	​


S2: 
⟨
𝑠
⟩
 
𝐼
 
𝑙
𝑜
𝑣
𝑒
 
𝑑
𝑒
𝑒
𝑝
 
𝑙
𝑒
𝑎
𝑟
𝑛
𝑖
𝑛
𝑔
 
⟨
/
𝑠
⟩
⟨s⟩ I love deep learning ⟨/s⟩

𝑃
(
𝐼
∣
⟨
𝑠
⟩
)
=
2
/
3
P(I∣⟨s⟩)=2/3

𝑃
(
𝑙
𝑜
𝑣
𝑒
∣
𝐼
)
=
1
P(love∣I)=1

𝑃
(
𝑑
𝑒
𝑒
𝑝
∣
𝑙
𝑜
𝑣
𝑒
)
=
1
/
2
P(deep∣love)=1/2

𝑃
(
𝑙
𝑒
𝑎
𝑟
𝑛
𝑖
𝑛
𝑔
∣
𝑑
𝑒
𝑒
𝑝
)
=
2
/
2
=
1
P(learning∣deep)=2/2=1

𝑃
(
⟨
/
𝑠
⟩
∣
𝑙
𝑒
𝑎
𝑟
𝑛
𝑖
𝑛
𝑔
)
=
1
/
2
P(⟨/s⟩∣learning)=1/2

𝑃
(
𝑆
2
)
=
2
3
⋅
1
⋅
1
2
⋅
1
⋅
1
2
=
1
6
P(S2)=
3
2
	​

⋅1⋅
2
1
	​

⋅1⋅
2
1
	​

=
6
1
	​


✅ Model prefers S1 because 
1
/
3
>
1
/
6
1/3>1/6.

B) Zero-probability problem

MLE:

𝑃
(
𝑛
𝑜
𝑜
𝑑
𝑙
𝑒
∣
𝑎
𝑡
𝑒
)
=
0
12
=
0
P(noodle∣ate)=
12
0
	​

=0

This is a problem because if any one bigram probability is 0, then the entire sentence probability becomes 0, which breaks probability comparisons and makes perplexity blow up / become undefined.

C) Laplace smoothing (Add-1)

Given: vocab size 
𝑉
=
10
V=10, total count after “ate” is 12, and count(ate,noodle)=0:

𝑃
(
𝑛
𝑜
𝑜
𝑑
𝑙
𝑒
∣
𝑎
𝑡
𝑒
)
=
0
+
1
12
+
10
=
1
22
P(noodle∣ate)=
12+10
0+1
	​

=
22
1
	​


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
