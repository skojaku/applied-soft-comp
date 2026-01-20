---
title: "Sprint Project Datasets"
---

# Recommended Datasets for Sprint Projects

This document lists the recommended datasets to provide for each sprint project.

## Module 1: The Tidy Data Escape Room

**Dataset:** WHO Tuberculosis Data (messy version)
- **Source:** World Health Organization Global Tuberculosis Report
- **Download:** [WHO TB Data](https://www.who.int/teams/global-tuberculosis-programme/data)
- **Why it works:** Contains merged header rows, country codes mixed with names, year columns that should be rows, age/sex encoded in column names, and summary statistics mixed with raw data
- **Alternative:** Excel files from government statistical agencies often have similar issues

**Preparation needed:**
- Download the Excel file with embedded formatting
- Keep the messy structure with merged cells and footnotes
- Provide as-is without cleaning

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
