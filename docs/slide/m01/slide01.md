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

# Advanced Topics in Network Science

Lecture 01: Introduction & Seven Bridges of Königsberg
Sadamori Kojaku

---

![bg right:100% width:70%](../enginet-intro-slide/enginet-01.png)

---

![bg right:100% width:70%](../enginet-intro-slide/enginet-02.png)

---

![bg right:100% width:70%](../enginet-intro-slide/enginet-03.png)

---

![bg right:100% width:70%](../enginet-intro-slide/enginet-04v2.png)


---

# Course Overview

- **Instructor:** Sadamori Kojaku (幸若完壮)
- **Email:** skojaku@binghamton.edu
- **Office Hours:** Tue & Thu 14:30-16:30
- **Course Website:** https://skojaku.github.io/adv-net-sci


---

# What is Network Science?
- What is ***Network***?
- Why do should we care about ***Network***?
- What is ***Network Science***?


![bg right:50% width:80%](https://github.com/skojaku/adv-net-sci/blob/gh-pages/_images/connected-component.jpg?raw=true)

---

# Find networks around you!

🦁🐘🐒 [My zoo of networks](https://skojaku.github.io/adv-net-sci/intro/zoo-of-networks.html) 🐼🦒🦓


---

# Is it just a branch of graph theory?

• 📐 Graph Theory:
  - Focuses on structured graphs (trees, grids, regular graphs)
  - Emphasizes mathematical properties

• 🌐 Network Science:
  - Studies complex networks in real-world systems
  - "Complex" ≠ "Complicated"
  - Seeks simple laws to explain seemingly intricate structures

---

# How is it different from data science?

• 📊 **Data Science: 1 + 1 = 2**
  - Often assumes independence between data points (i.i.d.)
  - Focuses on extracting insights from structured data

• 🌐 **Network Science: 1 + 1 > 2**
  - Embraces dependencies between entities
  - Recognizes that real-world systems are often interconnected
  - Analyzes how these connections influence system behavior

---

# Course Objectives

We will:
- 📊 Analyze networks
- 🧠 Learn key concepts
- 🤖 Apply AI to networks

After this course, you'll be able to:
- 📚 Understand network science papers
- 💻 Do advanced network analysis
- 🔬 Design network research
- 🔗 Connect Systems Science and networks


---

# Philosophy of Learning in this course

[![bg right:50% width:90% Drive: The surprising truth about what motivates us](https://img.youtube.com/vi/RQaW2bFieo8/0.jpg)](https://www.youtube.com/watch?v=RQaW2bFieo8 "Drive: The surprising truth about what motivates us")
https://www.youtube.com/watch?v=RQaW2bFieo8

---

![bg width:80% right:100%](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_NZxWrYf1VvFyG5we1DtkTZOkbsVkbUFtAg&s)

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
  - Proposal: Sept 29; Paper: Dec 4; Presentations: Dec 8

- Requirements 📋
  - Apply concepts to real problem 🌍
  - Analyze network dataset 🔬
  - Show course integration 🧠
  - Clear presentation 🗣️

---

# Example Project 01

![](sci-topic-net.png)

---

# Example Project 02

![width:100%](ecog.png)

---

# Example Project 03

Tesla Supercharger Network

![bg right:50% width:90%](super-charger.png)

---

# Exam

- 📚 Final exam on all topics (weight: 30%)
- 📅 During exam week (Dec 9-13)
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

- 📅 Roughly bi-weekly (weight: 20%)
- 💻 Coding exercises
- 🤖 Autograded assignments
- 🏁 Deadline: before final exam
- 🔄 Unlimited attempts until correct

---

# Lecture note

- 📚 [Interactive Jupyter book](https://skojaku.github.io/adv-net-sci)
- 💻 Run code directly on the page
  - ⏳ First-time loading may take 2-3 mins
- 🔄 Or download as Jupyter notebook
  - ☁️ Use on cloud (Google Colab, Kaggle) or locally
  - 📦 Install packages from `environment.yml` for local use
  - See [The course GitHub repo](https://github.com/skojaku/adv-net-sci/) for details

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

# Before we start
What motivates you to take this course (if you want to)?

[![Drive: The surprising truth about what motivates us](https://img.youtube.com/vi/u6XAPnuFjJc/0.jpg)](https://www.youtube.com/watch?v=u6XAPnuFjJc "Drive: The surprising truth about what motivates us")
https://www.youtube.com/watch?v=u6XAPnuFjJc

~8:23

---

# M01: Seven Bridges of Königsberg

![bg right:50% width:90% Königsberg Bridges](https://99percentinvisible.org/app/uploads/2022/02/bridges-with-water.png)

---

# The Königsberg Bridge Puzzle 🌉

![bg right:50% width:90% Königsberg Bridges](https://99percentinvisible.org/app/uploads/2022/02/bridges-with-water.png)

- 18th century puzzle in Königsberg, Germany 🇩🇪
- City had 7 bridges connecting 2 islands and mainland 🏙️
- **Challenge**: Find a route that crosses each bridge exactly once 🚶‍♂️

---

# Find a route that crosses each bridge exactly once 🚶‍♂️

How would you approach this problem?

![bg right:60% width:90%](https://physics.weber.edu/carroll/honors_images/bridges_of_konigsberg.jpg)

---

# Euler's Brilliant Solution 🧠

<img src="https://lh3.googleusercontent.com/-CYxppcJBwe4/W2ndkci9bVI/AAAAAAABX-U/K6SNM8gAhg0oNsnWNgQbH3uKNd5Ba10wwCHMYCw/euler-graph-bridges2?imgmax=1600" style="width: 100%; max-width: none; margin-bottom: 20px;">


- 🏙️ Simplified city to network of landmasses and bridges
- 🔗 Focused on connections, not layout



---


# Pen and Paper Exercise 📝

- Let's work on a pen-and-paper [exercise](http://estebanmoro.org/pdf/netsci_for_kids/the_konisberg_bridges.pdf) 📄
- Let's form a group of 3-4 people and discuss the solution together.

---

# Euler's Solution 🧠

- 🧮 Euler considered: even vs odd edge nodes
- 💡 Key insights: Even - enter/leave k times, Odd - one edge left
- 🏆 **Euler's Theorem**: Path exists if all even degree or two odd degree
- 🌉 Königsberg: All odd degree, no Euler path

![bg right:40% width:90%](../../lecture-note/figs/labeled-koningsberg.jpg)

---

# Aftermath: The Bridges' Fate 🏙️💣

- 🇷🇺 During World War II, Soviet Union bombarded Königsberg
- 💥 Two of the seven bridges were destroyed
- ✅ Ironically, this destruction made an Euler path possible!

![bg right:50% width:80%](../../lecture-note/figs/seven-bridge-bombared.png)


---

# 💻 Coding Time: Networks in Code! 🌐

[Let's represent networks with Python!](https://skojaku.github.io/adv-net-sci/m01-euler_tour/how-to-code-network.html) 🐍


---


# Key Takeaways

- Networks are powerful tools for modeling complex systems
- Euler's path problem: a foundation of graph theory and network science
- Python for network analysis

---

# Any questions?