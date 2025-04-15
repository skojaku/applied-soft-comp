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
    margin: 1em 0; /* Adjusted margin */
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

![bg right:40% opacity:0.8](https://images.squarespace-cdn.com/content/v1/5c44ef01620b85081a6abe38/1641768423976-QL7IXYNAGQN0I0WAR955/Inception_poster.jpg?format=1000w) <!-- Example AlexNet Diagram -->

---

<!-- Slide 1 -->
# After AlexNet 🚀

*   AlexNet (2012) proved deep ConvNets work for large-scale image recognition (ILSVRC winner).
*   **The Big Question:** How do we build *even better* networks? What drives performance 🤔?
*   Initial Hypothesis: *Maybe just go *deeper*?
*   Two major contenders from ILSVRC 2014 who explored this: **VGG** and **GoogLeNet**.

![bg right:40% width:100%](https://www.zdnet.com/a/img/resize/cbdfcc9ffe02c07ec17d656be49e670a55e467ec/2025/03/20/1fff3c66-1148-433b-859b-e53ca710522c/u-of-toronto-2013-hinton-krizhevsky-sutskever.jpg?auto=webp&width=1280) <!-- Example AlexNet Diagram -->

---

<!-- Slide 2 -->
# VGG: Betting on Depth and Simplicity 🏛️

*   From *Karen Simonyan & Andrew Zisserman* (Visual Geometry Group - VGG, Oxford)
*   **Core Question:** How does network *depth* impact performance?
*   **Strategy:** Isolate the effect of depth by keeping everything else extremely *simple* and *uniform*.
*   Challenged the notion (in 2014) that very deep networks were too difficult to train. VGG's success paved the way.

![bg right:25% width:100%](https://www.robots.ox.ac.uk/~vgg/assets/img/vgg_logo.png) <!-- VGG Logo -->

---

# VGG Architecture

![bg right:60% width:100%](https://miro.medium.com/v2/resize:fit:1400/1*NNifzsJ7tD2kAfBXt3AzEg.png)

---

<!-- Slide 3 -->
# The Uniform 3x3 Strategy 🧱

*   **Radical Simplicity:** Exclusively uses *3x3 filters* (stride 1, padding 1) and *2x2 Max Pooling* (stride 2) for downsampling.

**Why only 3x3 filters?**
1.  Smallest size capturing spatial context (up/down, left/right, center).
2.  **Stacking Benefit:**:
  *  `Stack of two 3x3` -> 5x5 receptive field
  *  `Stack of three 3x3` -> 7x7 receptive field

![bg right:45% width:100%](https://miro.medium.com/v2/resize:fit:1200/1*k97NVvlMkRXau-uItlq5Gw.png)

---

<!-- Slide 4 -->
# A Deep, Uniform Stack 🏗️

*   **Input:** 224x224x3 image
*   **Structure:** Repeating blocks of 3x3 convs + MaxPool.
*   **Channel Doubling:** `64 -> 128 -> 256 -> 512 -> 512`
*   **Spatial Reduction:** `224 -> 112 -> 56 -> 28 -> 14 -> 7` (Creates **"pyramid structure"**)
*   **Ends with Large Fully Connected Layers:** (e.g., 4096, 4096, 1000 for ImageNet)

![bg right:40% width:100%](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png) <!-- VGG16 Arch Diagram -->

---

<!-- Slide 5 -->
# 🤔 Thinking Point: Design Trade-offs

<div class="admonition question">

VGG uses *only* 3x3 filters for simplicity and to study depth.

**Question:** What might be the potential *disadvantages* of strictly using only one small filter size throughout such a deep network?

*(Hint: Think about features at different scales in an image).*

</div>

---

<div class="admonition answer">
💡 Answer: Using only 3x3 filters in VGG has several disadvantages:

1. **No Multi-Scale Processing**: VGG can't process different scales simultaneously within a layer.
2. **Computational Depth**: Requires many layers to achieve large receptive fields, increasing computational cost.

The uniform approach, while elegant, isn't optimal for capturing diverse feature scales in natural images compared to architectures with explicit multi-scale processing.
</div>

<!-- Note: The citation style {footcite}`...` won't render footnotes in standard Marp, but is kept for consistency with the source notes. You might need a Marp plugin or post-processing for actual footnotes. -->

---

<!-- Slide 6 -->
# VGG: Results ✅

*   **Success:** Excellent ILSVRC 2014 performance (Runner-up Class., Winner Localization).
*   **Confirmed:** Depth is crucial.
*   **Legacy:** Simple design, strong baseline, very popular for transfer learning.

---

# Other Enhancements

*   **Key Technique: VGG-style Data Augmentation:**
    *   Resized images to *multiple scales* (e.g., 256px, 384px heights) *before* random 224x224 cropping.
    *   Learned features robust to scale variations.

![bg right:50% width:100%](https://cvml-expertguide.net/wp-content/uploads/2021/08/e72850b7f9960fbbd9d51f636963baec.png)
*<small>Image: cvml-expertguide.net</small>*

---

<!-- Slide 7 -->
# The VGG Catch: Elephant in the Room 🐘

*   **Major Drawback:** Very computationally expensive.
*   **~140 Million Parameters (VGG16):**
    *   The vast **majority (~102 Million!)**.
    *   This issue was later addressed.

![bg right:35%](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Heavy_Weight_Vector.svg/512px-Heavy_Weight_Vector.svg.png) <!-- Heavy weight icon -->

---


<div class="admonition question">

VGG16 has ~140 million parameters. We saw that stacked 3x3 convolutions are relatively parameter-efficient compared to larger filters.

**Question:** Looking at the VGG architecture again, where do you hypothesize the *vast majority* of these parameters are located?


</div>

![bg right:40% width:100%](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png) <!-- VGG16 Arch Diagram -->

---


<!-- Slide 9 -->
# GoogLeNet: "Going Deeper with Convolutions!"

*   From Google (**Szegedy et al.**, 2014). *inner ILSVRC 2014 Classification*.
  - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
*   **Core Idea:** Don't just go deeper, go *smarter*. Design a better, multi-scale building block – the **Inception Module**.
*   **Analogy:** Like viewing a painting up close (details) and from at distance (composition) *simultaneously*.

![bg right:40% width:100%](https://images.squarespace-cdn.com/content/v1/5c44ef01620b85081a6abe38/1641768423976-QL7IXYNAGQN0I0WAR955/Inception_poster.jpg?format=1000w) <!-- Google Logo -->

---

<!-- Slide 10 -->
# Key Idea: Multi-Scale Processing 🔭🔬

*   **Strategy (Naive):** Apply multiple filter sizes (**1x1, 3x3, 5x5**) *in parallel* on the same input. Also include parallel *Max Pooling*.
*   **Goal:** Capture fine details (1x1) and broader context (3x3, 5x5) simultaneously.
*   **Output:** Concatenate the channels from all branches.
*  **This is computationally expensive!🤔**

![bg right:50% width:100%](https://miro.medium.com/v2/resize:fit:1400/1*DKjGRDd_lJeUfVlY50ojOA.png)

---

<!-- Slide 11 -->
<div class="admonition question">

The "naive" Inception module applies 1x1, 3x3, 5x5 filters (and pooling) in parallel and concatenates the outputs.

**Question:** Imagine stacking several of these naive modules. Consider the number of *output channels* after concatenation at each layer. What is the major computational problem that arises?

*(Hint: Think about the input channels for the *next* module).*

</div>

![bg right:50% width:100%](https://miro.medium.com/v2/resize:fit:1400/1*DKjGRDd_lJeUfVlY50ojOA.png)

---

<div class="admonition answer">

The main problem is **Channel Explosion**. Concatenating outputs from all parallel branches creates many more output channels than inputs. When stacked, this makes 3x3 and 5x5 convolutions computationally intractable due to the rapidly growing feature dimensions.

</div>

---

<!-- Slide 12 -->
# Breakthrough: 1x1 Convolutions

*   Use **1x1 Convolutions *before*** the convolutions.
    -  "Cross Channel Down sampling" by Min Lin et al. 2013
*   **Dimensionality Reduction:** Reduces channel depth fed into larger filters.
    *   `Input (H x W x C_in) -> 1x1 Conv (K filters) -> Output (H x W x K)` where `K < C_in`.
*   **Approximating Sparsity:** Efficiently processes info across channels (high correlation) before spatial convs (lower correlation).
*   **Parameter Reduction:** Saves computation significantly vs. direct 3x3/5x5 on full convolution.

![bg right:40% width:100%](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*dNaikOfrGzUaJ2EzRIl4tw.png)
*<small>Image: 1x1 Convolution Bottleneck - miro.medium.com</small>*

---

<!-- Slide 13 -->
# The Final Inception Module v1 ✨

Combines multi-scale processing with 1x1 bottlenecks for efficiency:

1.  **1x1 Conv path** (Direct)
2.  **1x1 Conv (reduce) -> 3x3 Conv path**
3.  **1x1 Conv (reduce) -> 5x5 Conv path**
4.  **3x3 Max Pool -> 1x1 Conv path** (Projection after pooling)

**Output:** Concatenate feature maps from all 4 branches along the channel dimension.

![bg right:50% width:100%](https://www.researchgate.net/publication/338353898/figure/fig1/AS:870495645876225@1584553743305/nception-module-with-dimensionality-reduction-12.jpg)

---

<center>
<img src="https://production-media.paperswithcode.com/methods/GoogleNet-structure-and-auxiliary-classifier-units_CM5xsxk.png", width="100%">

</center>


---


<!-- Slide 14 -->
# Auxiliary Classifiers

- Small classifiers added partway through during *training*.
- **Purpose:** Combat vanishing gradients by injecting loss/gradient deeper.
- Removed during inference.


![](https://production-media.paperswithcode.com/methods/GoogleNet-structure-and-auxiliary-classifier-units_CM5xsxk.png)

---

# Global Average Pooling (GAP)

- Replaces heavy Fully Connected layers at the end (Lin et al., 2013).
- Averages each final feature map spatially to a single value (`H x W x C_out -> 1 x 1 x C_out`).
- Drastically cuts parameters & overfitting risk (compare to VGG's FC layers!).


![bg right:50% width:100%](https://www.guidetomlandai.com/assets/img/machine_learning/global_average_pooling.png)

---


<!-- Slide 15 -->
<div class="admonition question">

GoogLeNet used auxiliary classifiers to help gradients reach early layers during training, addressing the "vanishing gradient" problem in deep networks.

**Question:** Can you think of *other* techniques (that came later) designed to solve the same problem of training very deep networks effectively?

*(Hint: Think about how information could bypass layers or blocks).*

</div>

---

<!-- Slide 16 -->
# GoogLeNet: Winning Efficiently 🏆🍃

*   **Result:** Won ILSVRC 2014 Classification task.
*   **Remarkable Efficiency:**
    *   Only **~7 Million parameters** (~20x fewer than VGG16!).
    *   Faster inference, less memory required.
*   **Key Message:** Proved that sophisticated architectural design could achieve state-of-the-art results more efficiently than brute-force depth alone.

---


<!-- Slide 17 -->
# VGG vs. GoogLeNet: Head-to-Head 🥊

| Feature             | VGG16                                  | GoogLeNet (Inception v1)                 |
| :------------------ | :------------------------------------- | :--------------------------------------- |
| **Core Idea**       | Depth via **Uniformity**             | Efficiency via **Multi-Scale Blocks**    |
| **Key Innovation**  | Stacking simple **3x3 convs**        | **Inception Module**, 1x1 Bottlenecks, GAP |
| **Parameters**      | **~140 Million**                       | **~7 Million**                           |
| **Main Bottleneck** | Heavy **Fully Connected** Layers     | Module Design Complexity                 |
| **Strengths**       | Simplicity, Good for Transfer Learning | High Efficiency, State-of-the-Art Perf.  |
| **Weaknesses**      | Very Heavy, Slow Inference           | More Complex Architecture                |
| **ILSVRC '14 Class.** | Runner-Up                              | **Winner**                               |

---

<!-- Slide 19 -->
# Legacy and Evolution 📈

*   **VGG Legacy:**
    *   Confirmed value of **depth**.
    *   Became a standard **baseline** and feature extractor due to **simplicity**.
*   **GoogLeNet Legacy:**
    *   Showcased power of **efficient architectural design**.
    *   Introduced influential concepts: **Inception module, 1x1 bottlenecks, Global Average Pooling**.
*   **Evolution of Inception:** The core idea inspired further improvements:
    *   **Inception v2/v3:** Batch Normalization, Filter Factorization.
    *   **Inception v4 / Inception-ResNet:** Added Residual Connections.
    *   **Xception:** Pushed towards Depthwise Separable Convolutions ("Extreme Inception").

---

<!-- Slide 20 -->
# Conclusion: Key Takeaways  takeaways 🔑

*   Both VGG and GoogLeNet were landmark achievements in 2014, significantly advancing deep learning for vision.
*   **VGG:** Demonstrated the power of **depth** achieved through **architectural simplicity and uniformity**, despite high computational cost.
*   **GoogLeNet:** Showed that **clever architectural design** (Inception modules, 1x1 bottlenecks, GAP) could yield state-of-the-art performance with remarkable **computational efficiency**.
*   These contrasting philosophies (simplicity vs. engineered efficiency) heavily influenced subsequent network designs.
