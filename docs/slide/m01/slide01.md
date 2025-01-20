---
marp: true
theme: default
paginate: true
---

Check list
- [ ] Microphone turned on
- [ ] Zoom room open
- [ ] MyBinder launched
- [ ] Sound Volume on

---

# SSIE 419/519: Applied Soft Computing

Lecture 01: Introduction & Word Embeddings

Sadamori Kojaku

---

![bg right:100% width:70%](../enginet-intro-slide/enginet-01.png)

---

![bg right:100% width:70%](../enginet-intro-slide/enginet-02.png)

---

![bg right:100% width:70%](../enginet-intro-slide/enginet-03.png)


---

# Course Overview

- **Instructor:** Sadamori Kojaku (幸若完壮)
- **Email:** skojaku@binghamton.edu
- **Office Hours:** Tue & Thu 15:00-17:00
- **Lecture Note:** https://skojaku.github.io/applied-soft-comp
- **GitHub:** https://github.com/skojaku/applied-soft-comp

---

# Why Soft Computing?

---

![bg right:100%](https://dynaimage.cdn.cnn.com/cnn/c_fill,g_auto,w_1200,h_675,ar_16:9/https%3A%2F%2Fcdn.cnn.com%2Fcnnnext%2Fdam%2Fassets%2F220902164101-01-thtre-dopra-spatial-ai-generated-art.jpg)

---

![bg right:100%](https://ewscripps.brightspotcdn.com/dims4/default/cad701e/2147483647/strip/true/crop/1920x1008+0+36/resize/1200x630!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3.amazonaws.com%2Fdf%2F0e%2F701346564b12bb9b271f39a10c27%2Fai-created-artwork-wins-first-place-at-colorado-state-fair-artist-receives-criticism.jpg)

---

![bg width:500px](https://thispersondoesnotexist.com/)

---

# People that do not exist

[ThisPersonDoesNotExist.com](https://thispersondoesnotexist.com/)

![](https://i.gzn.jp/img/2019/02/17/this-person-does-not-exist/00.jpg)

---


![bg width:1300px](https://cdn.mos.cms.futurecdn.net/wZvgUjoXojFGK7AJvMq6T7-320-80.gif)

---

# The Challenge 🤔

- Traditional computing requires **explicit rules**
- Consider face recognition:
  - How do you measure eye spacing?
  - What defines nose shape?
  - Which features matter most?

---

# How is it possible?

---

# Classsical Paradigm (60s-80s)

- Expert System
  - Explicit rules
- Examples:
  - MYCIN (1976) for medical diagnosis
- Issues:
  - No adaptability
  - Need a lot of rules for complex tasks


![bg right:50% width:100%](https://media.licdn.com/dms/image/v2/D4D12AQGd_VYIGInIrw/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1665910586302?e=2147483647&v=beta&t=bMmgo2EgEmXUJJy6mZUGwzGk7WvFEHS6oXgibYsnVxM)

---

# Classsical Paradigm (90s-2000s)

- Statistical Learning
  - Allow some randomness in real-world data
  - Examples:
    - Support Vector Machine
    - Random Forest
- Issues:
  - Need a lot of data
  - Need a lot of rules for complex tasks

---

# Deep Learning (2010s-2020s) 🧠

- Learn patterns from examples
- Adapt to new situations
- No need for explicit rules
- Handle complexity naturally

![bg right:55% width:100%](https://miro.medium.com/v2/resize:fit:1000/1*63sGPbvLLpvlD16hG1bvmA.gif)


---

# Language Understanding: ChatGPT 💭

- Natural conversation abilities
- Multiple capabilities:
  - Poetry and creative writing
  - Technical explanations
  - Code assistance
  - Understanding context and humor

![bg right:40%](https://media4.giphy.com/media/0lGd2OXXHe4tFhb7Wh/200w.gif?cid=6c09b9529cv8dqb9z5sunzgl9y907w53z0d5bnisg90db85i&ep=v1_gifs_search&rid=200w.gif&ct=g)

---

# Scientific Breakthrough: AlphaFold 🧬

- Solved 50-year protein folding problem
- Near-perfect accuracy
- Accomplished in days what took months in labs
- Revolutionary impact on biology and medicine

![bg right:40%](https://cdn.prod.website-files.com/64934a44d8e5c1c0ad207f62/654d623563b7ee9877fc86b1_Tool%20logos.png)

---

# Game AI: The Divine Move 🎮

- AlphaGo vs Lee Sedol
- Famous "Move 37"
- Demonstrated creative thinking
- Showed AI can surpass human intuition

![bg right:40%](https://i.guim.co.uk/img/media/6e698b71e16ecdeb5aca1d040b9c819a069f8065/469_955_2908_1744/master/2908.jpg?width=700&quality=85&auto=format&fit=max&s=29285b458f410ec9debcaf061b68e9f1)

---

# Video Generation: Sora 🎬

- Text-to-video technology
- 60-second realistic videos
- Physics-accurate scenes
- Multiple moving elements
- Photorealistic quality

![bg right:40%](https://cdn.mos.cms.futurecdn.net/wZvgUjoXojFGK7AJvMq6T7-320-80.gif)

---

# Medical Diagnosis: AI Assistant 👨‍⚕️

- Surpasses human accuracy in cancer detection
- Reduces:
  - Missed cases
  - False alarms
- Augments doctor's capabilities

![bg right:40%](https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/12/04/00/cancer-screening.jpg)

---

# Autonomous Systems: Self-Driving 🚗

- Real-time sensor processing
- Faster than human reactions
- Constant alertness
- Improved safety in many conditions

![bg right:40%](https://bernardmarr.com/wp-content/uploads/2021/07/How-Tesla-Is-Using-Artificial-Intelligence-to-Create-The-Autonomous-Cars-Of-The-Future.jpg)

---

# Why This Matters 🌟

- Unprecedented pace of breakthroughs
- Solving decades-old problems
- Surpassing human capabilities
- Transforming multiple fields:
  - Healthcare
  - Transportation
  - Scientific research
  - Creative arts

**Join the revolution in applied soft computing!**

---

# What motivates you to take this course (if you want to)?

[![Drive: The surprising truth about what motivates us](https://img.youtube.com/vi/u6XAPnuFjJc/0.jpg)](https://www.youtube.com/watch?v=u6XAPnuFjJc "Drive: The surprising truth about what motivates us")
https://www.youtube.com/watch?v=u6XAPnuFjJc

~8:23


---

# Course Structure

"Don't think! Feeeeeel" - Bruce Lee

- 🎓 Lectures
- 🛠️ Hands-on exercises
- 📝 Weekly quizzes
- 💻 Biweekly coding assignments
- 🎓 Final project
- 📝 Exam

![bg right:50% width:80%](https://media1.tenor.com/m/-LDi5jsgk_8AAAAd/bruce-lee-dont-think.gif)


---

# Final Project 🎓

- Individual project (30% of grade) 📊
- Timeline 📅
  - 03/09: Project Proposal
  - 05/06: Project Presentation
  - 05/09: Project Final Paper
- Requirements 📋
  - Apply concepts to real problem 🌍
  - Show course integration 🧠
  - Clear presentation 🗣️

---

# Exam

- 📚 Final exam on all topics (weight: 30%)
- 📅 During exam week
- 📝 Theory + practical problems
- 🌍 Apply concepts to real scenarios
- 📚 Review sessions before exam


---

# Weekly Quiz on Brightspace

- 📊 Quizzes: A tool to identify misconceptions (weight: 20%)
- 🧠 Covers previous week's topics
- 🏁 Deadline: before final exam
- 🔄 Unlimited attempts until correct

---

# Assignment

- 📅 One assignment per module (weight: 20%)
- 💻 Coding exercises
- 🤖 Autograded assignments
- 🏁 Deadline: before final exam
- 🔄 Unlimited attempts until correct

---

# Lecture note

- 📚 [Interactive Jupyter book](https://skojaku.github.io/applied-soft-comp)
- 💻 Run code directly on the page
  - ⏳ First-time loading may take 2-3 mins
- 🔄 Or download as Jupyter notebook
  - ☁️ Use on cloud (Google Colab, Kaggle) or locally
  - 📦 Install packages from `environment.yml` for local use
  - See [The course GitHub repo](https://github.com/skojaku/applied-soft-comp/) for details

---

# Policy

- 📚 3-credit course: 6.5+ hours of work/week outside class
- 🤖 AI tools allowed for learning, but cite if used in assignments
- 💾 Back up all data and code (loss not an excuse for late work)
- ♿ Accommodations available for students with disabilities
- 🚫 Zero tolerance for academic dishonesty


---

# Questions?


---

# The Challenge: Teaching Computers Language 🧮

- Computers only understand numbers
- Words need to be translated into numerical form
- Early approach: One-hot encoding
  - Each word gets a unique binary vector
  - Example: cat → [1,0,0], dog → [0,1,0]
- Problem: No semantic meaning captured

---

# From Words to Numbers: TF-IDF 📊

- Term Frequency (TF):

  $$
  TF(t,d) = \frac{\text{count of term t in doc d}}{\text{total terms in doc d}}
  $$

- Inverse Document Frequency (IDF):

  $$
  IDF(t) = \log\frac{\text{total documents}}{\text{docs containing t}}
  $$

- Combined score:

  $$
  TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)
  $$

---

# The Distributional Hypothesis 🌐


                "You shall know a word by the company it keeps"
            ~Words appearing in similar contexts have similar meanings~

![Distribution Example](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Word-embeddings-model.png?ssl=1)


---

## Word2Vec: Neural Word Embeddings 🧠

Two approaches: CBOW (Continuous Bag of Words) and Skip-gram
1. CBOW: Predict center word from context
2. Skip-gram: Predict context from center word

![bg right:60% width:800px](https://kavita-ganesan.com/wp-content/uploads/skipgram-vs-cbow-continuous-bag-of-words-word2vec-word-representation.png)

---

# Model (Skip-gram)

$$
P(w_o|w_c) = \frac{\exp(v_{w_o}^T v_{w_c})}{\sum_{w \in V} \exp(v_w^T v_{w_c})}
$$

Where:
- $v_{w_c}$ is center word vector
- $v_{w_o}$ is output word vector
- $V$ is vocabulary

![bg right:50% width:600px](../../lecture-note/figs/word2vec.jpg)

---


# Model (CBOW)


$$
P(w_c|w_1,...,w_C) = \dfrac{\exp(v_{w_c}^T \bar{v})}{\sum_{w \in V} \exp(v_w^T \bar{v})},
$$

where $\bar{v} = \dfrac{1}{C}\sum_{i=1}^C v_{w_i}$ is the average of the context word vectors.

---


# Matrix Factorization Connection 🔢

Word2vec implicitly factorizes a Pointwise Mutual Information (PMI) matrix:

$$
M_{ij} = \log \dfrac{P(w_i,  w_j)}{P(w_j)P(w_j)}
$$

Properties:
- Low when words appear independently
- High when words co-occur frequently
- Similar effect to TF-IDF normalization

Word embeddings preserve PMI values:

$$
v_{w_i} ^\top v_{w_j} \approx M_{ij}
$$

Key insight: Words that frequently appear in similar contexts will have similar embeddings!

---

# GloVe: Global Vectors 🌐

GloVe explicitly factorizes PMI matrix:


$$
M_{ij} = \log \frac{P(w_i, w_j)}{P(w_i)P(w_j)}
$$

![](https://storage.googleapis.com/coderzcolumn/static/tutorials/artificial_intelligence/word_embeddings.jpg)

---

# FastText: Handling Subwords 🔄

- Use subword information
- Example: "where" → "<wh", "whe", "her", "ere", "re>"
- Handles out-of-vocabulary words
- Better for morphologically rich languages


![bg right:50% width:700px](https://kavita-ganesan.com/wp-content/uploads/fastText-vs.-Word2Vec.png)


---

# SemAxis: Understanding Word Relationships 📐

- Identify two semantically interpretable clusters
- SemAxis: An axis between the cluster centroids, e.g., good-bad, soft-hard
- Project word vectors onto the SemAxis

![SemAxis Example](https://raw.githubusercontent.com/ghdi6758/SemAxis/master/doc/images/semaxis_diagram.png)

---

# Bias in Word Embeddings ⚖️

Gender bias example:
- man : doctor :: woman : nurse
- he : programmer :: she : homemaker

![bg right:50% width:500px](https://lena-voita.github.io/resources/lectures/word_emb/papers/gender_bias-min.png)

---

# Doc2Vec: From Words to Documents 📄

Two models:
1. PV-DM (Distributed Memory)
2. PV-DBOW (Distributed Bag of Words)

![bg right:50% width:100%](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JPetbQHmG0NAbdQ08JSiMQ.png)

---

# Doc2Vec (PV-DM)

PV-DM probability:

$$
P(w_i|w_{i-k},...,w_{i-1},d) = \frac{\exp(u_{w_i}^T h)}{\sum_{w \in V} \exp(u_w^T h)}
$$

Where $h$ is either:
- Average: $h = \frac{1}{k+1}(v_d + \sum_{j=i-k}^{i-1}v_{w_j})$
- Concatenation: $h = (v_d, \sum_{j=i-k}^{i-1}v_{w_j})U$

---

# Doc2Vec (PV-DBOW)

PV-DBOW probability:

$$
P(w_i|d) = \frac{\exp(u_{w_i}^T v_d)}{\sum_{w \in V} \exp(u_w^T v_d)}
$$

Where $v_d$ is the document vector.

---

# Summary: Evolution of Word Embeddings 🎓

1. One-hot encoding → No semantics
2. TF-IDF → Document-level patterns
3. Word2Vec → Local context patterns
4. GloVe → Global patterns
5. FastText → Subword information
6. Doc2Vec → Document embeddings

---

# Applications and Future Directions 💡

- Machine Translation
- Document Classification
- Information Retrieval
- Sentiment Analysis
- Question Answering
- Text Generation
