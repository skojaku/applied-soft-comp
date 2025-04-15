---
marp: true
theme: default
paginate: true
---


<style>
/* @theme custom */

/* Base slide styles */
section {
    width: 1280px;
    height: 720px;
    padding: 40px;
    font-size: 28px;
}

/* Marp Admonition Styles with Auto Labels */
.admonition {
    margin: 0em 0;
    padding: 1em 1em 1em 1em;
    border-left: 4px solid;
    border-radius: 3px;
    background: #f8f9fa;
    position: relative;
    line-height: 1.5;
    color: #333;
}

/* Add spacing for the title */
.admonition::before {
    display: block;
    margin-bottom: 0.8em;
    font-weight: 600;
    font-size: 1.1em;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Note */
.note {
    border-left-color: #0066FF;
}

.note::before {
    content: "ℹ️ Note";
    color: #0066FF;
}

/* Question */
.question {
    border-left-color: #00A550;
}

.question::before {
    content: "🤔 Question";
    color: #00A550;
}

/* Intuition */
.intuition {
    border-left-color: #9B59B6;
}

.intuition::before {
    content: "💭 Intuition";
    color: #9B59B6;
}


/* Answer */
.answer {
    border-left-color: #0066FF;
}

.answer::before {
    content: "💡 Answer";
    color: #0066FF;
}

/* Tip */
.tip {
    border-left-color: #00A550;
}

.tip::before {
    content: "💡 Tip";
    color: #00A550;
}

/* Important */
.important {
    border-left-color: #8B44FF;
}

.important::before {
    content: "📢 Important";
    color: #8B44FF;
}

/* Warning */
.warning {
    border-left-color: #CD7F32;
}

.warning::before {
    content: "⚠️ Warning";
    color: #CD7F32;
}

/* Caution */
.caution {
    border-left-color: #FF3333;
}

.caution::before {
    content: "🚫 Caution";
    color: #FF3333;
}

/* Two-Column Layout for Marp Slides */
/* Basic column container */
.columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1rem;
    height: calc(100% - 2rem); /* Account for margin-top */
}

/* Left column */
.column-left {
    grid-column: 1;
    color: inherit;
    min-width: 0; /* Prevent overflow */
}

/* Right column */
.column-right {
    grid-column: 2;
    color: inherit;
    min-width: 0; /* Prevent overflow */
}

/* Optional: Equal height columns with top alignment */
.columns-align-top {
    align-items: start;
}

/* Optional: Center-aligned columns */
.columns-align-center {
    align-items: center;
}

/* Optional: Different column width ratios */
.columns-40-60 {
    grid-template-columns: 40fr 60fr;
}

.columns-60-40 {
    grid-template-columns: 60fr 40fr;
}

.columns-30-70 {
    grid-template-columns: 30fr 70fr;
}

.columns-70-30 {
    grid-template-columns: 70fr 30fr;
}

/* Ensure images scale properly within columns */
.column-left img,
.column-right img {
    max-width: 100%;
    height: auto;
}

/* Optional: Add borders between columns */
.columns-divided {
    column-gap: 2rem;
    position: relative;
}

.columns-divided::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    border-left: 1px solid #ccc;
    transform: translateX(-50%);
}

/* Fix common Markdown elements inside columns */
.columns h1,
.columns h2,
.columns h3,
.columns h4,
.columns h5,
.columns h6 {
    margin-top: 0;
}

.columns ul,
.columns ol {
    padding-left: 1.5em;
}

/* Ensure code blocks don't overflow */
.columns pre {
    max-width: 100%;
    overflow-x: auto;
}
</style>


# Image Processing 🧠

<center>
<img src="https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F511d51bd1d1ec3b7155250bf7e53cfa6cb52f215-1339x503.png&w=3840&q=75" width="80%">
</center>

---

# Early Success of LeNet

- LeNet demonstrated the first success of neural networks in image classification
- Despite its success, there was a 20-year period of *winter* of neural networks
- ...until AlexNet's breakthrough in 2012

![bg right:50% width:100%](https://vitalab.github.io/article/images/lenet/a384.gif)

---

# The Winter of Neural Networks ❄️


- Neural networks couldn't scale beyond shallow architectures
- **Vanishing gradients** made deeper models untrainable
- **Compute bottlenecks** limited experimental depth
- Classical methods (SVMs, random forests) outperformed on benchmarks

By the 2000s, neural nets were largely abandoned for practical tasks.


![bg right:30% width:100%](https://szeliski.org/Book/imgs/SzeliskiBookFrontCover.png)

[See the "Table of Contents" of the 2011 version of a well-known computer vision textbook](https://www.cs.bilkent.edu.tr/~s.rahimzadeh/SzeliskiBook_20100903_draft.pdf)


---

# Other reasons for the winter of neural networks 🧠🚫

- Statistical learning emphasized **interpretable, convex models**
- Neural nets seen as empirical, black-box, unprincipled
- **Peer review and grants** favored theory-heavy approaches
- Tooling was fragmented, poorly documented, and hard to scale

![bg right:50% width:100%](https://miro.medium.com/v2/resize:fit:960/0*UW8gLZMDbBHjYV0o.png)
![bg right:50% width:100%](https://target.scene7.com/is/image/Target/GUEST_98a339fc-8f28-4767-affb-ce7d21667100)

---

# Feature Engineering Era 🛠️


- Common pipeline:
  1. **Handcrafted features**: SIFT, HOG, LBP
  2. Dimensionality reduction: PCA, LDA
  3. Classifier: SVM, Random Forest, k-NN

- Image Recognition $\simeq$ Feature Engineering problem

- If you have good features, you can improve image recognition performance

![bg right:45% width:100%](https://cdn.techscience.cn/ueditor/files/cmc/TSP-CMC-68-2/TSP_CMC_16467/TSP_CMC_16467/Images/fig-4.png/mobile_webp)

---

# Two missing pieces for deep learning 🧩


---

# Data 🧩

- Before 2006: Progress in computer vision was bottlenecked by small, narrow datasets.
- Fei-Fei Li (when she was at U. Illinois Urbana-Champaign) proposed a radical idea: **build a dataset as large and varied as the real world**, called **ImageNet**.
- She led a team to build ImageNet, a dataset of 14 million images across 20,000 categories.

![bg right:50% width:100%](https://pi.tedcdn.com/r/pe.tedcdn.com/images/ted/fbada01990f86f5afa850cc23a0259fec091f929_2880x1620.jpg?u%5Br%5D=2&u%5Bs%5D=0.5&u%5Ba%5D=0.8&u%5Bt%5D=0.03&quality=82c=1050%2C550&w=1050)

---

# How ImageNet was built 🗂️🖼️

1. **Choose Categories**
   - Used **WordNet** hierarchy for organization (e.g., "Siberian husky" → "dog" → "animal" → "entity")
2. **Collect Images**
   - Gathered images from web searches (~1000-2000 images per category)
3. **Verify Images**
   - Amazon Mechanical Turk workers checked images (simple yes/no verification process)
4. **Clean Dataset**
   - Removed duplicates and poor quality images

![bg right:40% width:100%](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTN4LrxH3IemJEt67e-Nicu7SX9SbUjoAPbkjslEjJ-7xwrcmdYzPy9dVHksuNHF76J9iw&usqp=CAU)

---

# From Dataset to Benchmark – ILSVRC 📊

- 2010: ImageNet team launched **ILSVRC**, a yearly classification challenge.
- Focused subset: 1,000 categories, 1.2M images.
- Defined the **state of the art** in vision.

![bg right:50% width:100%](https://i.namu.wiki/i/bdGEA6xDd5_JkfesHRWM26dtb9iHEMA-GcYb1NO7KyybQORUwy9pLrWarYpL53qLnPofD4g3S4cBAUM6RMqsdA.webp)

---

# [👉 ILSVRC is very tough ...](https://image-net.org/challenges/LSVRC/2012/analysis/#cmp_pascal_div)


---

# Only 2% improvement in one year

![bg right:70% width:100%](./before-alex-net.png)


---

# In 2012, one algorithm sets the deep learning revolution

---

## AlexNet's Performance Leap (2012)

- **16.4% top-5 error**, down from ~25% the previous year
- A neural network with 8 layers, 60M parameters, trained on two GPUs


![bg right:50% width:100%](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F1937562f4ac2507386e0a1965602544f697bb439-665x419.png&w=1920&q=75)


---

# AlexNet: A Breakthrough in Deep Learning

![bg right:50% width:100%](https://www.zdnet.com/a/img/resize/cbdfcc9ffe02c07ec17d656be49e670a55e467ec/2025/03/20/1fff3c66-1148-433b-859b-e53ca710522c/u-of-toronto-2013-hinton-krizhevsky-sutskever.jpg?auto=webp&width=1280)

- Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton (2012)
- Sparked the modern deep learning revolution


---

# Key architectural and training innovations
  1. ReLU activation
  2. Dropout regularization
  3. Local response normalization
  4. GPU-parallel training


---

# Innovation 1: ReLU Activation 🧮


- ReLU was introduced in 2010 by Hinton's group
- **Sigmoid**:
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

  - Saturates for large $|x|$, slows learning
- **ReLU**:
    $$\text{ReLU}(x) = \max(0, x)$$
  - Linear for $x > 0$, zero otherwise
  - Preserves gradients, speeds up training

![bg right:50% width:100%](https://miro.medium.com/v2/resize:fit:474/1*HGctgaVdv9rEHIVvLYONdQ.jpeg)

---

# ReLU Variants – Leaky & PReLU

- Problem: ReLU units can **"die"** if $x$ stays negative
- Fix:
  - **Leaky ReLU**: small slope for $x \leq 0$
  - **PReLU**: slope $\alpha$ is learnable

$$
f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$

- [👉 Interactive visualization of activation functions](https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks/)

![bg right:50% width:100%](https://miro.medium.com/v2/resize:fit:1400/1*A_Bzn0CjUgOXtPCJKnKLqA.jpeg)


---

# Innovation 2: Dropout Regularization

- **Large networks overfit easily** without constraint
- Dropout: randomly deactivate neurons with probability $p$ during training
- At inference, all units active but scaled by $(1-p)$
- Encourages redundancy and robustness

![bg right:60% 100%](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/04/1IrdJ5PghD9YoOyVAQ73MJw.gif)

---

# Innovation 3: Local Response Normalization

## What is normalization?

- **Normalization ~ Standardization of the pixel values**

- Without normalizations, models may learn to rely on the scale of the pixel values.

- A simple normalization: The whitening of the pixel values

![bg right:50% width:100%](https://i.sstatic.net/bcwMn.png)

---


### **Local Response Normalization (LRN)**

- Channels indexed $1, 2, 3, \dots, C$
- Normalizes each activation by **neighboring channels**:

$$
b^i_{x,y} = \frac{\underbrace{a^i_{x,y}}_{\text{activation}}}{\underbrace{\left(\underbrace{k}_{\text{bias}} + \underbrace{\alpha}_{\text{scale}} \underbrace{\sum_{j = i - n/2}^{i + n/2} (a^j_{x,y})^2}_{\text{sum over nearby channels}}\right)}_{\text{normalization factor}} ^\beta}
$$

- Suppresses redundant activations and noise

![bg right:50% width:100%](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*kHj2cSjWCUXTaWHZPjZn6Q.png)

---


<div class="admonition note">
LRN was introduced in AlexNet but later fell out of use—BatchNorm proved more effective and efficient.
</div>

---


# AlexNet Architecture Overview 🧱

<div>
<div class="columns">
  <div class="column-left">
    <ol style="font-size: 0.9em;">
      <li><strong>Input</strong>: 224×224 RGB image</li>
      <li><strong>Conv1</strong>: 11×11, stride 4 → ReLU → LRN → MaxPool</li>
      <li><strong>Conv2</strong>: 5×5, stride 1 → ReLU → LRN → MaxPool</li>
      <li><strong>Conv3–5</strong>: 3×3 convolutions with ReLU</li>
    </ol>
  </div>
  <div class="column-right">
    <ol start="5" style="font-size: 0.9em;">
      <li><strong>MaxPool</strong> after Conv5</li>
      <li><strong>FC6</strong>: 9216 → 4096 → ReLU → Dropout</li>
      <li><strong>FC7</strong>: 4096 → 4096 → ReLU → Dropout</li>
      <li><strong>FC8</strong>: 4096 → 1000 → Softmax</li>
    </ol>
  </div>
</div>

<center>
<img src="https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F511d51bd1d1ec3b7155250bf7e53cfa6cb52f215-1339x503.png&w=3840&q=75" width="80%">
</center>
</div>

---

<div class="admonition question"> AlexNet splits its convolutional layers into two parallel paths. This is due to a hardware limitation at the time. What motivated this architectural choice? </div>

<center>
<img src="https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F511d51bd1d1ec3b7155250bf7e53cfa6cb52f215-1339x503.png&w=3840&q=75" width="80%">
</center>

---

<div class="admonition answer">
Memory Constraints: GPUs in 2012 had limited memory (~3GB). Splitting the network reduced the memory burden on each GPU.

Parallel Computation: Running two paths in parallel accelerated training by distributing the workload.

</div>

<center>
<img src="https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F511d51bd1d1ec3b7155250bf7e53cfa6cb52f215-1339x503.png&w=3840&q=75" width="80%">
</center>

---

- Left: ILSVRC-2010 test images and the five labels considered most probable by the model
- Right: ILSVRC-2010 test images in the first column. The remaining columns show the six training images that produce feature vectors in the last hidden layer with the smallest Euclidean distance from the feature vector for the test image.

<img src="https://neurohive.io/wp-content/uploads/2018/10/Capture-16.jpg" width="100%">

---

# Legacy of AlexNet 🌍

- Paved the way for deep learning to dominate CV and beyond
- Inspired architectures: VGG, GoogLeNet, ResNet
- Shifted the field from handcrafted to learned representations
- Validated empirical, data-driven progress over hand-tuned features