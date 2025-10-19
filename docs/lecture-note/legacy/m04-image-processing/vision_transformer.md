# Vision Transformers

![](https://miro.medium.com/v2/resize:fit:700/0*Rtb7Jt6378xfe6Z1.png)

What if we could capture not just the local features in images (like corners, edges, or textures) but the entire global context all at once? Could that help a model better understand complex scenes and relationships between objects? Vision Transformers (ViT) attempt exactly that by leveraging **self-attention**, a mechanism originally popularized in Natural Language Processing.

## The Genesis of Vision Transformers

### Conceptual Foundation

*Why were Vision Transformers developed, given that Convolutional Neural Networks (CNNs) already excel in computer vision tasks?*

CNNs have been the cornerstone of computer vision for years, particularly good at capturing local patterns through convolutional filters. However, they can struggle to efficiently capture **global context** and **long-range dependencies**. In scenarios where relationships between objects spread across an entire image become crucial (e.g., understanding crowd scenes or satellite imagery), this limitation can be significant.

Meanwhile, the **Transformer architecture** (from the paper "Attention Is All You Need") revolutionized NLP by modeling long-range dependencies in sequential data. This success inspired researchers to ask: *Could the same self-attention mechanism help models 'see' the entire image at once, instead of focusing on small, local regions?*

```{figure} https://www.researchgate.net/publication/361733806/figure/fig3/AS%3A1173979050057729%401656909825527/Operation-of-CNN-and-ViT.ppm
:name: fig-cnn-vit
:width: 500px
:align: center

Comparison of the receptive field of CNNs and Vision Transformers. CNN has a local receptive field constrained by the convolutional filters, while ViT has a global receptive field, allowing it to capture long-range dependencies between different parts of the image.

```

```{note}
**Historical Context**

- **CNN Dominance (2010s)**: CNNs (e.g., AlexNet, VGG, ResNet) drove huge leaps in image classification and object detection.
- **Transformer Breakthrough (2017)**: In NLP, Transformers replaced recurrent architectures (LSTMs, GRUs) for tasks like machine translation.
- **ViT Emerges (2020)**: Google researchers introduced the idea of applying pure Transformers to image patches, showing excellent results on large-scale image datasets.
```

## The Vision Transformer (ViT) Architecture

*How do we adapt an NLP-centric Transformer to handle 2D image data?*

In **Vision Transformers**, an image is first split into a grid of small, equally sized patches—commonly $16 \times 16$ pixels each. Each patch is **flattened** and fed into a linear layer that creates a higher-dimensional embedding. You can think of each patch embedding as analogous to a "word embedding" in NLP.

```{figure} https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F7a096efc8f3cc40849ee17a546dc0e685da2dc73-4237x1515.png&w=3840&q=75
:name: fig-vit-patch
:width: 500px
:align: center

The process of splitting an image into patches and feeding them into a Vision Transformer. Image taken from [Pinecone](https://www.pinecone.io/learn/series/image-search/vision-transformers/).
```


```{note}
**Why Patches Instead of Pixels?**

- Handling each pixel independently would create a massive sequence (e.g., a 224x224 image has 50176 pixels!).
- Using patches reduces sequence length substantially and preserves local spatial structure.
```

### 2.2 Positional Encodings

Because Transformers are order-agnostic, we add **positional encodings** to each patch embedding. These encodings help the model understand the position of each patch in the original image grid.

### 2.3 Transformer Encoder

The sequence of patch embeddings (plus positional encodings) goes through a **Transformer encoder**, consisting of:
- **Multi-Head Self-Attention**: Allows each patch to attend to others, learning both local and global image features.
- **Feed-Forward Layers (MLP blocks)**: Expands and contracts the hidden dimension to add non-linear transformations.

### 2.4 Classification Head

Typically, the **[CLS] token** (a special token prepended to the sequence) serves as the global representation. After passing through all encoder layers, it goes to a lightweight classification head (a small MLP) to predict the output class.

```{note}
**Mathematical Foundation (Simplified)**

Self-attention for a single head can be described as:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- $Q, K, V$ are linear projections of the input (patch embeddings).
- $d_k$ is the dimension of $K$.
- Multi-head attention runs this process in parallel with different learnable projections.
```

---

## 3. Types of Attention Mechanisms

Even though Vision Transformers generally use **multi-head self-attention**, research has explored variations:

1. **Stochastic “Hard” Attention**: Focuses on a subset of patches while ignoring others.
2. **Deterministic “Soft” Attention**: Assigns weights to all patches.
3. **Multi-Head Attention**: Employs multiple attention heads to learn different aspects (textures, edges, shapes) simultaneously.

```{note}
**Implementation Insight**

You can vary the attention mechanism to strike different balances between computational cost and representational capacity. Hard attention can be more efficient but trickier to train.
```

---

## 4. Implementation Example

Let’s walk through a simplified code snippet using **Hugging Face Transformers** to classify images with a Vision Transformer. This gives a concrete look at how to build upon these theoretical concepts in practice.

```{code-cell} ipython3
# A minimal ViT classification example with Hugging Face

!pip install transformers
!pip install torch
!pip install torchvision

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import requests

# Load a pre-trained Vision Transformer model
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# Load an example image
url = "https://images.unsplash.com/photo-1524820353064-7141fab1f5b6"
image = Image.open(requests.get(url, stream=True).raw)

# Prepare the image
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

# Print the predicted label
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

```{tip}
- **Data Augmentation**: When training from scratch on smaller datasets, apply heavy data augmentation (random crops, flips, color jitter) to avoid overfitting.
- **Transfer Learning**: Leverage pre-trained weights from large datasets and then fine-tune on your target task for improved performance.
- **Batch Size**: ViTs can be memory-heavy; consider smaller batch sizes and gradient accumulation if GPU memory is limited.
```

![](https://via.placeholder.com/600x300)
[Figure: A schematic visualization of patches being extracted from an image and fed into a Vision Transformer pipeline.]

---

## 5. Advancements in Vision Transformer Architectures

Could we make ViTs more data-efficient, faster, or better at capturing hierarchical features?

### 5.1 Improved Training and Architectures

- **DeiT (Data-efficient Image Transformers)**: Introduces a distillation step to improve data efficiency, making ViTs competitive with CNNs on smaller datasets.
- **Model Soups**: Averages predictions from multiple ViT models to harness their individual strengths for higher accuracy.

### 5.2 Hierarchical and Hybrid Approaches

- **Swin Transformer**: Processes images in a hierarchical manner using non-overlapping patches at different resolutions, improving scalability to arbitrary image sizes.
- **CaiT (Cross-Attention Image Transformer)**: Uses cross-attention between different patch groups to capture more complex relationships.
- **CSWin Transformer**: Adopts a cross-shaped window self-attention pattern to optimize the balance between spatial coverage and computational cost.
- **FDViT**: Employs flexible downsampling layers for smoother feature map reductions, improving efficiency and classification accuracy.

```{note}
**Common Misconception**

It’s tempting to think ViTs automatically solve all the limitations of CNNs. However, they still require careful tuning, large datasets (or pre-training), and thoughtful architecture decisions to perform at their best.
```

---

## 6. Strengths and Weaknesses

When do Vision Transformers shine, and where do they falter?

| Feature             | Convolutional Neural Networks (CNNs)      | Vision Transformers (ViTs)                       |
|---------------------|-------------------------------------------|--------------------------------------------------|
| **Architecture**    | Convolutional + pooling + MLP             | Pure Transformer with self-attention             |
| **Input Processing**| Processes entire image as is              | Splits image into patches (tokens)               |
| **Global Context**  | Emerges in deeper layers                  | Captured from the start across all patches       |
| **Data Requirements**| Perform well with moderate data          | Often require very large datasets or pre-training|
| **Compute Cost**    | Usually lower, localized ops              | Higher due to self-attention on all patches      |
| **Performance**     | Excellent with well-tuned architectures   | Excels on large-scale data, state-of-the-art SOTA|

### Key Advantages

- **Global Context**: The self-attention mechanism can integrate information from all patches simultaneously.
- **Scalability**: ViTs shine on large datasets, often surpassing CNNs.
- **Reduced Inductive Bias**: They learn more general representations since they are not hard-coded to look for local spatial features like CNNs.

### Main Limitations

- **Data-Hungry**: Tend to overfit on small datasets; methods like DeiT and heavy augmentation help.
- **High Computational Cost**: Each patch attends to all others, which can be expensive for high-resolution images.
- **Interpretability**: Visualizing attention maps is possible, but can still be less intuitive than CNN feature maps.

---

## 7. Real-World Applications

How are Vision Transformers being used beyond simple image classification?

- **Object Detection & Image Segmentation**: Self-attention helps capture relationships among objects scattered across the scene.
- **Medical Imaging**: Identifying tumors in X-rays or segmenting organ boundaries in MRI scans.
- **Remote Sensing**: Analyzing satellite imagery for deforestation tracking or disaster management.
- **Action Recognition in Videos**: Extended to video frames, ViTs can learn complex spatiotemporal patterns.
- **Multi-Modal Tasks**: Works well with textual data (e.g., image captioning, visual question answering).
- **Autonomous Driving**: Understanding global context on the road is critical for safe navigation.
- **Anomaly Detection**: Identifying unusual patterns or defects in manufacturing lines.

```{note}
**Real-World Use Case**

In **medical imaging**, ViTs can better spot anomalies by focusing on subtle global context differences in scans. This can help radiologists detect diseases in early stages and potentially save lives.
```

---

## 8. Future Directions

- **Enhanced Efficiency**: Model compression, pruning, and improved patch strategies aim to reduce computational overhead.
- **Smaller Dataset Training**: More advanced self-supervision, distillation, and data-augmentation techniques are being developed to tackle data limitations.
- **Interpretability**: Research on attention visualization tools and explanations is growing, aiming to make ViTs more transparent.
- **New Domains**: From multi-modal reasoning to video analysis, ViTs are expanding across countless tasks in AI.

```{note}
**Performance Optimization**

- Distillation from large teacher ViTs or even CNNs can help small ViTs converge faster with less data.
- Layer-wise learning rate decay and progressive resizing of patches are common training tricks.
```

---

## 9. Reflection and Exercises

1. **Reflection**: Why do you think ViTs require large datasets to perform optimally, and how might transfer learning mitigate this requirement?
2. **Exercise**: Implement a fine-tuning script for a ViT on a smaller dataset (e.g., CIFAR-10). Try various data augmentation strategies. Compare results with a CNN baseline.
3. **Advanced Exploration**: Experiment with Swin Transformer or CSWin Transformer. Observe how hierarchical patching or specialized window attention changes performance and training speed.

---

## References

1. Vision Transformers - The Future of Computer Vision! [ResearchGate]
2. Introduction to Vision Transformers | Original ViT Paper Explained [aipapersacademy.com]
3. Deploying Attention-Based Vision Transformers to Apple Neural Engine [machinelearning.apple.com]
4. Vision Transformers (ViT) in Image Recognition: Full Guide [viso.ai]
5. Vision Transformers, Explained. A Full Walk-Through of Vision… [Towards Data Science]
6. From Transformers to Vision Transformers (ViT): Applying NLP Models to Computer Vision [Medium]
7. A Comprehensive Study of Vision Transformers in Image Classification Tasks [arXiv]
8. Introductory guide to Vision Transformers [Encord]
9. Efficient Training of Visual Transformers with Small Datasets [NeurIPS Proceedings]
10. FDViT: Improve the Hierarchical Architecture of Vision Transformer [ICCV 2023]
11. Vision Transformers vs. Convolutional Neural Networks (CNNs) [GeeksforGeeks]
12. Vision Transformers vs CNNs at the Edge [Edge AI Vision]
13. What is a Vision Transformer (ViT)? Real-World Applications [SJ Innovation]
14. Vision Transformer: An Introduction [Built In]
15. Mastering Vision Transformers with Hugging Face [Rapid Innovation]
16. Vision Transformer (ViT) - Hugging Face [huggingface.co]
17. Top 10 Open Source Computer Vision Repositories [Encord]
18. yhlleo/VTs-Drloc: Efficient Training of Visual Transformers [GitHub]
19. Vision Transformers for Image Classification: A Comparative Survey [MDPI]
20. BMVC 2022: How to Train Vision Transformer on Small-scale Datasets? [GitHub]
21. Vision Transformer: What It Is & How It Works [V7 Labs]

---

```{note}
**Key Takeaway**

Vision Transformers offer a fresh approach to image understanding by modeling global relationships among patches from the get-go. As architectures and training strategies evolve, they stand poised to become foundational building blocks in next-generation computer vision systems.
```