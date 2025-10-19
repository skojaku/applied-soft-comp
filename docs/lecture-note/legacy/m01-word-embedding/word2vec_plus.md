---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# GloVe and FastText: Building on Word2Vec's Foundation

word2vec inspires a lot of follow-up work. Here we will introduce two notable ones: **GloVe** and **FastText**.


## GloVe: Looking at the Big Picture

GloVe approaches word embeddings from a fundamentally different perspective than Word2Vec. While Word2Vec learns incrementally by scanning through text with small context windows, predicting words from their neighbors, GloVe takes a more global approach by analyzing the entire corpus's word co-occurrence patterns at once.

### Key Differences from Word2Vec

While Word2Vec learns incrementally by predicting context words through a neural network architecture, GloVe takes a more direct approach through matrix factorization. It explicitly models relationships between all word pairs at once, allowing it to capture global patterns that Word2Vec's local window approach might miss. GloVe's mathematical foundation as a matrix factorization model, similar to LSA but with improved weighting, makes its training objective more interpretable and connects it naturally to classical statistical methods.

The key insight behind GloVe is that the ratio of co-occurrence probabilities carries meaningful information. Let's look at a concrete example:

```{code-cell} ipython3
import gensim.downloader as api
import numpy as np
from tabulate import tabulate

# Load pre-trained GloVe vectors
glove = api.load('glove-wiki-gigaword-100')

# Demonstrate word relationships
word_pairs = [
    ('ice', 'water', 'steam'),
    ('king', 'queen', 'man'),
    ('computer', 'keyboard', 'screen')
]

def analyze_relationships(model, word_pairs):
    results = []
    for w1, w2, w3 in word_pairs:
        # Calculate cosine similarities
        sim12 = model.similarity(w1, w2)
        sim23 = model.similarity(w2, w3)
        sim13 = model.similarity(w1, w3)
        results.append([f"{w1}-{w2}", sim12])
        results.append([f"{w2}-{w3}", sim23])
        results.append([f"{w1}-{w3}", sim13])
        results.append(['---', '---'])

    print(tabulate(results, headers=['Word Pair', 'Similarity'],
                  floatfmt=".3f"))

print("Analyzing word relationships in GloVe:")
analyze_relationships(glove, word_pairs)
```

These similarities demonstrate how GloVe captures semantic relationships. Notice how related word pairs (like 'ice-water' and 'water-steam') have higher similarities than less related pairs (like 'ice-steam').

Let's also look at how GloVe handles analogies:

```{code-cell} ipython3
# Demonstrate analogies
analogies = [
    ('king', 'man', 'queen', 'woman'),  # gender relationship
    ('paris', 'france', 'london', 'england'),  # capital-country
    ('walk', 'walked', 'run', 'ran')  # verb tense
]

def test_analogy(model, word1, word2, word3, expected):
    result = model.most_similar(
        positive=[word3, word2],
        negative=[word1],
        topn=1
    )[0]
    print(f"{word1} : {word2} :: {word3} : {result[0]}")
    print(f"Expected: {expected}, Confidence: {result[1]:.3f}\n")

print("Testing analogies in GloVe:")
for w1, w2, w3, w4 in analogies:
    test_analogy(glove, w1, w2, w3, w4)
```

GloVe's global co-occurrence statistics allow it to perform well on analogy tasks because it captures the overall structure of the language, not just local patterns. This is particularly evident in:

1. **Semantic Relationships**: GloVe can identify pairs of words that appear in similar contexts across the entire corpus
2. **Syntactic Patterns**: It captures grammatical relationships by learning from how words are used globally
3. **Proportional Analogies**: The famous "king - man + woman = queen" type relationships emerge naturally from the co-occurrence patterns

The success of GloVe in these tasks demonstrates the value of its matrix factorization approach and global statistics, complementing the incremental learning strategy of Word2Vec. Both approaches have their strengths, and understanding their differences helps in choosing the right tool for specific NLP tasks.

```{note}
The ability to capture global statistics makes GloVe particularly good at analogies, but it requires more memory during training than Word2Vec because it needs to store the entire co-occurrence matrix.
```



## FastText: Understanding Parts of Words

FastText took a different approach to improving word embeddings. Its key insight was that words themselves have internal structure that carries meaning. Consider how you understand a word you've never seen before, like "unhelpfulness." Even if you've never encountered this exact word, you can understand it by recognizing its parts: "un-" (meaning not), "help" (the root word), and "-fulness" (meaning the quality of).

FastText implements this insight through several key mechanisms:

1. **Subword Generation**: Break words into character n-grams
   - Example: "where" → "<wh", "whe", "her", "ere", "re>"
   - The < and > marks show word boundaries

2. **Vector Creation**:
   - Each subword gets its own vector
   - A word's final vector is the sum of its subword vectors
   - This allows handling of new words!

Let's see FastText in action:

```{code-cell} ipython3
from gensim.models import FastText

# Train a simple FastText model
sentences = [
    ["the", "quick", "brown", "fox", "jumps"],
    ["jumping", "is", "an", "action"],
    ["quick", "movement", "requires", "energy"]
]

model = FastText(sentences, vector_size=100, window=3, min_count=1)

# FastText can handle words not in training
print("Similar words to 'jumped' (not in training):")
print(model.wv.most_similar('jumped'))
```

## When to Use Each Approach

The choice between these models often depends on your specific needs:

### GloVe is Better For:
- Capturing broad thematic relationships
- Working with a fixed vocabulary
- Tasks involving analogies and word relationships

### FastText Excels When:
- You expect new or misspelled words
- Working with morphologically rich languages
- Handling rare words is important

## Modern Impact

The innovations from GloVe and FastText haven't been forgotten in modern NLP. Today's large language models like BERT and GPT incorporate insights from both approaches:
- They use subword tokenization (like FastText)
- Their attention mechanisms capture both local and global patterns
- They can generate context-dependent representations

The evolution from Word2Vec to GloVe to FastText shows how different perspectives on word representation have contributed to our understanding of language. Each model made unique contributions while building on previous insights, laying the groundwork for today's more sophisticated approaches.

```{note}
Think of these models as different ways of looking at the same problem:
- Word2Vec is like learning from conversations
- GloVe is like analyzing entire books at once
- FastText is like understanding word roots and affixes
```