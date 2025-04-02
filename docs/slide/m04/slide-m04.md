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

---

# Image Processing Fundamentals: Edge Detection 🖼️


<img src="https://media.geeksforgeeks.org/wp-content/uploads/20240616211411/Screenshot-(85).webp" alt="Edge Detection" width="800px" style="display: block; margin: 0 auto;">

---

# Basic Image Processing 📸

- Image = 2D matrix of pixel values
- Each pixel represents brightness/color
- Example grayscale image:

![bg right:40%](https://ai.stanford.edu/~syyeung/cvweb/Pictures1/imagematrix.png)

$$
X = \begin{bmatrix}
10 & 10 & 80 & 10 & 10 & 10 \\
10 & 10 & 80 & 10 & 10 & 10 \\
10 & 10 & 80 & 10 & 10 & 10 \\
10 & 10 & 80 & 10 & 10 & 10 \\
10 & 10 & 80 & 10 & 10 & 10 \\
10 & 10 & 80 & 10 & 10 & 10
\end{bmatrix}
$$

---

# Convolution: Spatial Domain 🔄

- Slide kernel over image
- Multiply and sum values
- Example kernel (vertical edge detection):
- [Demo](https://setosa.io/ev/image-kernels/)

$$
K = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}
$$

![bg right:50% width:140%](https://miro.medium.com/v2/resize:fit:1400/1*D6iRfzDkz-sEzyjYoVZ73w.gif)

---

# Convolution is Complicated 😬

**Example**:
Suppose we have an image $X$ and a kernel $K$ as follows:

$$
\begin{aligned}
X &= \begin{bmatrix}
X_1 & X_2 & X_3 & X_4 & X_5 & X_6
\end{bmatrix} \\
K &= \begin{bmatrix}
K_1 & K_2 & K_3
\end{bmatrix}
\end{aligned}
$$

The convolution is given by

$$
X * K = \sum_{i=1}^{6} X_i K_{7-i}
$$

Or equivalently,

$$
X * K = \begin{bmatrix}
X_1 K_3 + X_2 K_2 + X_3 K_1 & X_2 K_3 + X_3 K_2 + X_4 K_1 & X_3 K_3 + X_4 K_2 + X_5 K_1 & X_4 K_3 + X_5 K_2 + X_6 K_1
\end{bmatrix}
$$

---

# Let's make it simpler using **the convolution theorem**!

## What is the convolution theorem?
Suppose two functions $f$ and $g$ and their Fourier transforms $F$ and $G$. Then,
  $$
  \underbrace{(f * g)}_{\text{convolution}} \leftrightarrow \underbrace{(F \cdot G)}_{\text{multiplication}}
  $$

The Fourier transform is a one-to-one mapping between $f$ and $F$ (and $g$ and $G$).

But what is the Fourier transform 🙃?

---

# Fourier Transform: The Basics 🌊

Transform a signal from:
- Time/Space domain ➡️ Frequency domain

Key Concept:
- Any signal can be decomposed into sum of sine and cosine waves
- Each wave has specific frequency and amplitude

![bg right:50% width:100%](https://devincody.github.io/Blog/post/an_intuitive_interpretation_of_the_fourier_transform/img/FFT-Time-Frequency-View_hu24c1c8fe894ecd0dad24174b2bed08c9_99850_800x0_resize_lanczos_2.png)

---

# 2D Fourier Transform 📊

2D Fourier Transform decomposes image into sum of *2D* waves.


$$\mathcal{F}(X)[h, w] = \sum_{k=0}^{H-1} \sum_{\ell=0}^{W-1} X[k, \ell] \cdot \underbrace{e^{-2\pi i \left(\frac{hk}{H} + \frac{w\ell}{W}\right)}}_{2D \text{ wave}}$$

For image $X$ with size $H \times W$.

<img src="https://i0.wp.com/thepythoncodingbook.com/wp-content/uploads/2021/08/fourier_transform_4a-356506902-1630322213734.png?resize=739%2C359&ssl=1" alt="2D Fourier Transform" width="700px" style="display: block; margin: 0 auto;">


---

# Edge Detection in Frequency Domain 🔍

Original Image ➡️ Fourier Transform ➡️ Apply Filter ➡️ Inverse Transform

```python
# Convert to frequency domain
FX = np.fft.fft2(img_gray) # Image
FK = np.fft.fft2(kernel_padded) # Kernel

# Multiply
filtered = FX * FK

# Convert back
result = np.real(np.fft.ifft2(filtered))
```


---

# JPEG Compression 📸

1. Divide image into 8x8 blocks
2. Apply Discrete Cosine Transform (similar to Fourier)
3. Quantize frequencies
   - Keep low frequencies
   - Discard high frequencies
4. Encode efficiently

Benefits:
- Smaller file size
- Maintains visual quality
- Exploits human visual perception

![bg right:40% width:100%](https://upload.wikimedia.org/wikipedia/commons/2/23/Dctjpeg.png)

---

# Coding Exercise

[Coding Exercise](https://github.com/skojaku/applied-soft-comp/blob/main/notebooks/image-processing-01.ipynb)


---

#

1. Divide image into 8x8 blocks
2. Apply Discrete Cosine Transform (similar to Fourier)
3. Quantize frequencies
   - Keep low frequencies
   - Discard high frequencies
4. Encode efficiently

Benefits:
- Smaller file size
- Maintains visual quality
- Exploits human visual perception

![bg right:40% width:100%](https://upload.wikimedia.org/wikipedia/commons/2/23/Dctjpeg.png)

---

# Coding Exercise

[Coding Exercise](https://github.com/skojaku/applied-soft-comp/blob/main/notebooks/image-processing-01.ipynb)


---

# From Hand-crafted to learnable kernels 🧠

---


# Reading Handwritten Digits ✏️🔢

In the 1980s, banks needed machines to **automatically read ZIP codes and checks** written by hand.

But handwriting is messy.

- Different styles, slants, shapes
- No two people write a "5" the same way
- Ink is noisy, paper is imperfect

Traditional rule-based programs were **too rigid** and **too brittle** for the real world.

Can a machine learn to make sense of such variation?

![bg right:30% width:100%](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSKj0d9vXffDLexgIihrnAek-Nr4ukBlaKFZMdyOaja_buKnuNFFLo-cLKoNelav2aauh0&usqp=CAU)

---

# Limited computational power 🖥️🐢

Now imagine solving this task using computers from the early 1990s:

- CPU speed: ~33 MHz
- RAM: ~4 MB
- No GPUs, no large datasets, no fancy toolkits

And yet, a simple neural network did it.

**Question:**
How can we build a system that can understand handwritten digits—**robustly, accurately, and fast enough to be deployed at scale**?

![bg right:30% width:100%](https://upload.wikimedia.org/wikipedia/commons/9/9f/Ibm_pcjr_with_display.jpg)

---

# A remarkable solution ~ Convolutional Neural Networks 🧠

---

# LeNet: The First Convolutional Neural Network 🧠

- Invented by Yann LeCun in 1988
- Multiple iterations: LeNet-1, LeNet-5, etc.
- Achieved 99.2% accuracy on MNIST
- Deployed commercially for reading checks

![bg right:30% width:100%](https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Yann_LeCun_-_2018_%28cropped%29.jpg/220px-Yann_LeCun_-_2018_%28cropped%29.jpg)

## LeNet-1: [Demo](https://www.youtube.com/watch?v=nQtbB9sd81A)

---

# How LeNet Works ⚙️🧩

LeNet learns a **hierarchy of visual patterns**:

## [👉Visualization](https://tensorspace.org/html/playground/lenet.html)


1. **<span style="color:blue">Convolution (C1)</span>**: Detects basic edges and textures using 4 filters (5×5)

2. **<span style="color:red">Pooling (S2)</span>**: Reduces dimensions from 24×24 to 12×12

3. **<span style="color:blue">Convolution (C3)</span>**: Combines basic features into complex patterns with 12 filters

4. **<span style="color:red">Pooling (S4)</span>**: Further reduces to 4×4

5. **<span style="color:black">Fully Connected</span>**: Aggregates them into classification probabilities (digits 0-9)

The breakthrough? All features are **learned automatically** from raw pixels, not hand-engineered.

---

# Let's dive into the details of **<span style="color:blue">Convolution Layers</span>** 🧠

---


# What Is Convolution? 🔍📸

- Slide a small **kernel** over the image
- Multiply-and-sum with the overlapping pixels
- Output: a **feature map** that highlights the pattern

![width:90%](https://anhreynolds.com/img/cnn.png)

---

# Convolution in Math 🧮🧠

For a grayscale image and 3×3 kernel:

$$
(I * K)_{i,j} = \sum_{m=0}^{2} \sum_{n=0}^{2} I_{i+m,j+n} \cdot K_{m,n}
$$

- $I$: input image
- $K$: kernel
- Output: new pixel at position $(i,j)$ that encodes the match strength

---

## Why Convolution Works ⚙️✨

✅ Detects features **regardless of position** (translation equivariance)
✅ Parameter efficient than fully connected NN
  - Example: 1024x1024 image with a 3x3 kernel:
    - Convolution: 3x3 = 9 parameters
    - Fully connected: 1024x1024x1024 = 1024^3 ~ 10^9 parameters 🤯

![width:80%](https://miro.medium.com/v2/resize:fit:1400/1*NoAQ4ZgofpkK6esl4sMHkA.png)

---

## Multi-Channel Convolution

Real images aren't grayscale—they have **multiple channels** (e.g., RGB).

- **Channel**: A dimension of the input image (e.g., RGB image has 3 channels)
- A kernel has **one slice per channel**
- **Sum the results with offsets** for each channel, producing a single-channel feature map


<div align="center">
<img src="https://d2l.ai/_images/conv-multi-in.svg" width="70%">
</div>

---

<div class="admonition question">

**Question**: How many parameters are in a convolutional layer with 1 input channel, 1 output channel, and 3×3 kernels?
</div>

<br>

<div class="admonition answer">

**Answer**:
$\text{Parameters} = \underbrace{1}_{\text{input channel}} \times \underbrace{1}_{\text{output channel}} \times \underbrace{3 \times 3}_{\text{kernel size}} + \underbrace{1}_{\text{bias}} = 10$

The +1 represents the bias term. Each output channel has one bias parameter, so with 1 output channel, we have 1 bias parameter in addition to the 9 weight parameters from the 3×3 kernel.

</div>

---

<div class="admonition question">

**Question**: How many parameters are in a convolutional layer with 64 input channels, 128 output channels, and 3×3 kernels?

</div>

---

<div class="admonition answer">

**Answer**:
$\text{Parameters} = \underbrace{64}_{\text{input channels}} \times \underbrace{128}_{\text{output channels}} \times \underbrace{3 \times 3}_{\text{kernel size}} + \underbrace{128}_{\text{bias}} = 73,856$

For a convolution layer with $C_{in}$ input channels, $C_{out}$ output channels, and $K \times K$ kernel size, the total number of parameters is:

$$\text{Parameters} = C_{in} \times C_{out} \times K \times K + C_{out}$$

Where:
- $C_{in} \times C_{out} \times K \times K$ are the weights
- $C_{out}$ are the bias terms (one per output channel)


</div>


---

# Receptive Field: Benefit of hieracical convolution 👁️📏

Each output pixel is influenced by a **region of the input image**—this is the receptive field.

- Early layers: small receptive fields (edges, textures)
- Deeper layers: larger fields (shapes, objects)
- More depth = more context

![bg right:50% width:100%](https://www.researchgate.net/publication/316950618/figure/fig4/AS:11431281212123378@1702542797323/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks.tif)

---


<div class="column-right admonition question">

**Question**: How many parameters are needed for a neuron to have a 5×5 receptive field?

**Option 1**: Single 5×5 convolution

**Option 2**: Two stacked 3×3 convolutions

</div>

![bg right:50% width:100%](https://www.researchgate.net/publication/316950618/figure/fig4/AS:11431281212123378@1702542797323/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks.tif)

---

<div class="admonition answer">

**Answer**:

**Option 1**: Single 5×5 convolution = 25 parameters (5×5)

**Option 2**: Two stacked 3×3 convolutions = 18 parameters (3×3 + 3×3)

Using stacked smaller convolutions is more parameter-efficient while achieving the same receptive field!

</div>

---

<div class="column-right admonition question">

**Question**: What is the receptive field of the last layer in the following

**Option 1**: Two stacked 2x2 convolutions

**Option 2**: Three stacked 3x3 convolutions

</div>

![bg right:50% width:100%](https://www.researchgate.net/publication/316950618/figure/fig4/AS:11431281212123378@1702542797323/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks.tif)

---

<div class="admonition answer">

**Answer**:

**Option 1**: Two stacked 2x2 convolutions = 3x3 receptive field
- First layer: 2x2
- Second layer: Each output pixel sees 2x2 area of first layer outputs
- Total: 3x3 receptive field

**Option 2**: Three stacked 3x3 convolutions = 7x7 receptive field
- First layer: 3x3
- Second layer: Each output pixel sees 3x3 area of first layer outputs (5x5 total)
- Third layer: Each output pixel sees 3x3 area of second layer outputs (7x7 total)

</div>

---

# Stride: Step Size 🚶‍

Stride determines how far the kernel moves at each step.

- **Stride = 1** → dense, overlapping coverage
- **Stride = 2** → skips positions, downsamples the image
- Larger stride = fewer outputs, lower resolution

![bg right:50% width:90%](https://miro.medium.com/v2/resize:fit:1400/1*X22-wmPcir4y5VoeqDyvWg.gif)

---

# Padding

Convolution shrinks images because boundary pixels don't have enough neighbors.

- **Valid padding**: No padding, smaller output
  - 5×5 input + 3×3 kernel → 3×3 output
- **Same padding**: Adds zeros around borders to maintain size
  - 5×5 input + 3×3 kernel + padding → 5×5 output
- Padding preserves spatial information and prevents shrinking

![bg right:45% width:100%](https://svitla.com/uploads/ckeditor/2024/Math%20at%20the%20heart%20of%20CNN/image_930660943761713546482755.gif)

---

# [👉 Visualization of Convolution in "What is a Convolutional Neural Network?"](https://miro.medium.com/v2/resize:fit:1400/1*X22-wmPcir4y5VoeqDyvWg.gif)

---
# Output Size Formula 🧮📏

To compute the output dimension $O$:

$$
O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

- $W$: input size
- $K$: kernel size
- $P$: padding
- $S$: stride

---

# Pooling Layers: Shrinking While Preserving 👣📉

Pooling reduces the spatial size while keeping important features.

- Common types: **Max pooling**, **Average pooling**
- Operates on small windows (e.g., 2×2, 3×3)
- Makes the network more robust to small shifts and noise

<div style="text-align: center;">
    <img src="https://epynn.net/_images/pool-01.svg" width="100%">
</div>

---

# Max vs. Average Pooling 📊

### Max Pooling:
$$
P_{i,j} = \max_{m,n} F_{i+m,j+n}
$$

### Average Pooling:
$$
P_{i,j} = \frac{1}{K^2} \sum_{m,n} F_{i+m,j+n}
$$

- Max: keeps the strongest activation
- Average: smooths the features

Pooling introduces **translation invariance** and reduces computation.

---

# Strided Convolution vs. Pooling 🤔

Some modern architectures skip pooling entirely.

| Feature              | Pooling Layer     | Strided Convolution   |
|----------------------|-------------------|------------------------|
| Parameters           | None              | Learnable              |
| Invariance           | Built-in          | Learned                |
| Flexibility          | Fixed operation   | Tunable                |

<div class="admonition note">

Pooling is simple and effective—but strided convs offer more control.

</div>

---



# How LeNet Works (Revisit) ⚙️🧩

LeNet learns a **hierarchy of visual patterns**:

## [👉Visualization](https://tensorspace.org/html/playground/lenet.html)


1. **<span style="color:blue">Convolution (C1)</span>**: Detects basic edges and textures using 4 filters (5×5)

2. **<span style="color:red">Pooling (S2)</span>**: Reduces dimensions from 24×24 to 12×12

3. **<span style="color:blue">Convolution (C3)</span>**: Combines basic features into complex patterns with 12 filters

4. **<span style="color:red">Pooling (S4)</span>**: Further reduces to 4×4

5. **<span style="color:black">Fully Connected</span>**: Aggregates them into classification probabilities (digits 0-9)

The breakthrough? All features are **learned automatically** from raw pixels, not hand-engineered.

---

# Figure from the original paper. Can you now read this?

<div style="text-align: center;">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*ge5OLutAT9_3fxt_sKTBGA.png" width="100%">
</div>


![bg right:50% width:100%]()

---

# [LeNet in PyTorch](https://github.com/skojaku/applied-soft-comp/blob/master/notebooks/lenet.ipynb)
