---
title: "Sprint Project Datasets"
---

# Recommended Datasets for Sprint Projects

This document lists the recommended datasets to provide for each sprint project.

## Module 1: The Tidy Data Escape Room

**Dataset:** Gapminder World Data (intentionally messified)
- **Source:** [Gapminder](https://www.gapminder.org/data/)
- **Clean version variables:** `country`, `continent`, `year`, `lifeExp`, `pop`, `gdpPercap`
- **Why it's the gold standard:** This dataset perfectly demonstrates tidy data's versatility. Students can create temporal views, correlation plots, and distribution comparisons just by changing which columns map to x, y, color, and size—without reshaping the data at all.

**The Messy Version You'll Provide:**

Transform the clean Gapminder data into a nightmare by introducing these tidy data violations:

1. **Wide format years:** Convert years to columns (`lifeExp_1952`, `lifeExp_1957`, etc.)
2. **Merged headers:** Combine `continent` and `country` into single cells with formatting
3. **Metadata in cells:** Add footnotes like "* estimated" directly in data cells
4. **Summary rows:** Insert continental averages mixed with country data
5. **Implicit missing values:** Leave some cells blank instead of explicit `NA`
6. **Ambiguous column names:** Use `LE` instead of `lifeExp`, `Pop` and `population` interchangeably

**Preparation script:**

```python
import pandas as pd
import numpy as np

# Load clean Gapminder
df = pd.read_csv('gapminder.csv')

# Create messy version
# 1. Pivot to wide format for life expectancy
messy = df.pivot_table(
    values='lifeExp',
    index=['country', 'continent'],
    columns='year'
)
messy.columns = [f'lifeExp_{col}' for col in messy.columns]

# 2. Add summary rows
for continent in df['continent'].unique():
    continent_data = df[df['continent'] == continent]
    summary_row = {f'lifeExp_{year}': continent_data[continent_data['year'] == year]['lifeExp'].mean()
                   for year in df['year'].unique()}
    summary_row['country'] = f'AVERAGE - {continent}'
    summary_row['continent'] = continent
    # Insert summary rows

# 3. Add metadata annotations (*, †, etc.)
# 4. Merge header cells in Excel export
# 5. Use inconsistent column naming

messy.to_excel('gapminder_messy.xlsx')
```

**The Payoff:**

Once students clean this into proper tidy format, they can demonstrate its power by creating:
1. **Temporal view:** Life expectancy evolution over time (line chart)
2. **Correlation view:** GDP vs. Life Expectancy (scatter with size=population)
3. **Distribution view:** Life expectancy by continent (boxplot)

## Module 2: The Ugly Graph Makeover

**Dataset:** Gapminder World Data (GDP, Life Expectancy, Population)
- **Source:** [Gapminder](https://www.gapminder.org/data/)
- **Why it works:** Rich dataset with temporal, geographic, and numeric dimensions allowing for diverse visualizations
- **Variables:** Country, Year, Life Expectancy, GDP per Capita, Population

**Ugly chart to provide:**
Create a misleading 3D pie chart or truncated bar chart showing GDP comparisons, or a line chart with dual y-axes that exaggerate differences. Students must identify the issues and create three honest alternatives.

## Module 3: Dashboard from Scratch

**Dataset:** NYC 311 Service Requests (sample)
- **Source:** [NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)
- **Why it works:** Multiple categorical variables (complaint type, borough, agency), timestamps for temporal analysis, geolocation data, and enough variety for interesting interactive filtering
- **Size:** Provide a sample of ~10,000-50,000 rows (manageable for 60 minutes)

**Alternative:** Kaggle's Superstore Sales dataset (smaller, easier to work with)

## Module 4: The Vibe-Check Classifier

**Dataset:** Reddit Post Titles from Multiple Subreddits
- **Source:** Use Reddit API or provide pre-collected sample
- **Subreddits:** Mix technical (r/programming, r/machinelearning) with casual (r/CasualConversation, r/AskReddit), professional (r/business) with creative (r/writing)
- **Why it works:** Clear variation across multiple semantic dimensions, perfect for finding surprising misclassifications
- **Size:** 500-1000 post titles

**Preparation:**
```python
# Example collection script
import praw
reddit = praw.Reddit(client_id='...', client_secret='...', user_agent='...')
# Collect top 200 posts from each of 5 subreddits
```

## Module 5: Adversarial Art Attack

**Dataset:** ImageNet Sample Images
- **Source:** [ImageNet](https://www.image-net.org/) or use torchvision's sample images
- **Provide:** 10-20 well-classified images of common objects (banana, dog, car, coffee mug, keyboard)
- **Why it works:** Pre-trained models on ImageNet make this straightforward, and everyday objects create humorous misclassifications

**Preparation:**
- Select images that ResNet-50 classifies correctly with >90% confidence
- Include diverse object categories for variety
- Ensure images are high quality (224x224 minimum after cropping)

## Module 6: The Network Saboteur

**Dataset:** Zachary's Karate Club Network
- **Source:** Built into NetworkX (`nx.karate_club_graph()`)
- **Why it works:** Classic social network with clear community structure, small enough to visualize completely, well-studied so instructors know the critical edges
- **Size:** 34 nodes, 78 edges
- **Critical edges:** Edges between nodes 0-3 bridge the two factions

**Alternative:** Les Misérables character co-occurrence network (also built into NetworkX)

**Preparation:**
```python
import networkx as nx
G = nx.karate_club_graph()
nx.write_edgelist(G, 'karate_club.txt')
```

## Module 7: The Les Misérables Identity Crisis

**Dataset:** Les Misérables Text + Co-occurrence Network
- **Text Source:** [Project Gutenberg](https://www.gutenberg.org/ebooks/135)
- **Network Source:** Built into NetworkX (`nx.les_miserables_graph()`)
- **Why it works:** Well-known story, rich character relationships, clear opportunities for semantic-structural gaps

**Preparation:**
1. Download the full text from Project Gutenberg
2. Extract the co-occurrence network from NetworkX
3. Provide character list to help with preprocessing

```python
import networkx as nx
G = nx.les_miserables_graph()
# Network has character names as nodes, co-occurrence frequency as weights
nx.write_edgelist(G, 'les_mis_network.txt', data=['weight'])
```

**Expected gaps:**
- Javert: Frequent textual mentions with Valjean but structurally distant
- Thénardiers: Structurally central but semantically peripheral
- Minor characters: High structural centrality but low semantic presence

## General Notes

- **Data Size:** Keep datasets small enough to work with in 60 minutes
- **Data Quality:** Pre-validate that datasets work with the kickstarter code
- **Documentation:** Provide a README with column descriptions and data sources
- **Testing:** Run through each sprint yourself before class to estimate timing
