# InceptionNet: Image Recognition CNN by Google (GoogLeNet)
Posted September 20, 2021 by Masaki Hayashi

## Table of Contents
1. What is InceptionNet [Overview]
   1.1 Article Structure
   1.2 Core Ideas and Key Points of Each Version
   1.3 Origin of InceptionNet's Name
2. v1 (GoogLeNet)
   2.1 Inception Module
      2.1.1 Multiple Size Path Parallelization
      2.1.2 Dimension Reduction for Lightweight Design
   2.2 Global Average Pooling
   2.3 Auxiliary Classifiers: Stabilizing Backpropagation to Early Layers
3. v2: Introduction of Batch Normalization
4. v3: Improvement of Inception Blocks
   4.1 Parameter Reduction through Factorization
   4.2 Effective Spatial Size Reduction
   4.3 Label Smoothing
5. Inception-v4 and Inception-ResNet
6. Summary

## 1. What is InceptionNet [Overview]
InceptionNet (Inception Network, also known as GoogLeNet) is a CNN (Convolutional Neural Network) architecture devised by Google's research team [Szegedy et al., 2015]. After InceptionNet v1, improved versions v2, v3, and v4 were successively released.

This article will introduce Inception v1 through v4 in order of their appearance, focusing on the important points of each version.

*Note (Important): This site consistently uses the unique term "InceptionNet." When searching for "Inception" alone, search results are dominated by the movie Inception (see Section 1.2), making it difficult to find this article directly. When searching for this article directly from Google or other search engines, please include "InceptionNet" in your query (when referring to specific versions, we use the standard notation like "Inception v3" without "Net").*

### 1.1 Article Structure
The following sections will introduce each version in order:

1. Section 1: Key points of each version (1.2) and origin of the name (1.3)
2. Section 2: Inception v1 (GoogLeNet) [CVPR2015]: Initial version
3. Section 3: Inception v2 (BN-Inception) [ICML2015]: Introduction of batch normalization
4. Section 4: Inception v3 [CVPR2016]: Improvement of Inception blocks
5. Section 5: Inception v4 and Inception-ResNet v1, v2 [AAAI 2017]: Improvements with residual connections
6. Section 6: Summary

### 1.2 Core Ideas and Key Points of Each Version
The central idea of InceptionNet (Inception v1 = GoogLeNet, Section 2) lies in "approximating and replacing each convolution layer with computationally efficient Inception blocks."

The Inception block (Section 2.1) is a block that uses "dimension reduction in the channel direction using 1x1 convolution layers" and "parallelization with multiple convolution sizes in 4 paths (=widening)." This allowed the use of blocks that were both expressive and computationally efficient as basic building units.

Inception v1 succeeded in significantly reducing computational costs through parameter reduction compared to previous CNN backbones like AlexNet, while improving expressiveness by making the network deeper (winning first place in ILSVRC 2014). While Inception v2 was proposed in the batch normalization paper with the addition of batch normalization, it had no other contributions (new proposals).

Next, Inception v3 [Szegedy et al., 2016b] evolved into a more efficient and accurate InceptionNet with the proposal of new Inception blocks that had better balance between width and depth (Section 4). Label smoothing was also an important contribution from the Inception v3 research.

Finally, the Inception v4 research [Szegedy et al., 2017] completed the final form of Inception-ResNet (v1, v2), which reached a 96-layer configuration through ResNet-style additions with residual connections (Section 5). As it combined the efficiency of Inception modules with the advantages of residual connections in optimizing large-scale models, Inception-ResNet was widely used afterward, particularly in scenarios "where model size was necessary," such as video recognition.

*Note: The use of multiple layers as a repeating unit is called a "block" in networks like ResNet and DenseNet. While Inception v1 and v3 papers call it an "Inception module," we will consistently refer to it as an "Inception block" here. The Google team called it "Inception module" in papers from Inception v1 through Inception v3, but in the Inception v4 paper, they unified the terminology with ResNet-style "block" and began calling it "Inception block."*

### 1.3 Origin of InceptionNet's Name
The network name "Inception" comes from the hit SF action movie Inception of that time (a favorite of the site administrator).

The movie title "Inception" means "beginning," named after the idiomatic phrase "inception of an idea." In English, "inception of an idea" refers to something that starts small but later develops into a bigger idea. In the movie Inception, the protagonists are a special operations group that secretly enters their target's lucid dreams, going into "deeper layers" of dreams. They attempt to "plant ideas" by acting in the dreams without being detected, going down to deep layers of subconscious. Hence, the movie was titled "Inception (of an idea)."

The movie Inception was a massive hit worldwide at the time, and the line "We need to go deeper" spoken by the protagonist (DiCaprio) to the wealthy target became popular as an image macro and internet slang in English-speaking regions. The authors of InceptionNet [Szegedy et al., 2015] named their network and paper after this movie's story of descending through multiple dream layers.

At the time, top researchers studying new CNN structures were competing to "improve image recognition accuracy by making CNNs deeper." They drew a parallel between the movie Inception's "dream within a dream (going deeper through multiple layers)" hierarchical meta-structure and their "meta-network structure = Inception module" utilizing the Network-in-Network [Lin et al., 2014] concept. Therefore, their paper was similarly titled "Going deeper with convolutions."

While this lengthy explanation shows how young researchers sometimes give (occasionally immature) names to their methods during the third AI boom, younger researchers should avoid imitating this as it tends to become embarrassing in retrospect.

## 2. v1 (GoogLeNet)
InceptionNet v1's network structure is "an architecture using Inception blocks instead of convolution layers" in an AlexNet-style serial CNN:

- conv1: Convolution layer - [7 x 7] kernel, stride=2
- pool1: Max pooling - [3 x 3] kernel, stride=2
- conv2: Convolution layer - [3 x 3] kernel, stride=2
- pool2: Max pooling - [3 x 3] kernel, stride=2
- 3a: Inception block
- 3b: Inception block
- pool3: Max pooling - [3 x 3] kernel, stride=2
- 4a: Inception block
- 4b: Inception block
- 4c: Inception block
- 4d: Inception block
- 4e: Inception block
- pool4: Max pooling - [3 x 3] kernel, stride=2
- 5a: Inception block
- 5b: Inception block > Output size: [7 × 7] x 1024
- pool5: Global average pooling - [7 x 7] kernel, stride=1 > Output size: [1 × 1] x 1024
- linear6: Fully connected layer: 1024 x 1000
- softmax: 1000-class probability distribution

### 2.1 Inception Module
The Inception block is the basic building unit that makes up InceptionNet. Within the Inception block, it performs (1) diverse size path parallelization (Figure 1-a, Section 2.1.1) and (2) dimension reduction for each path to achieve lightweight design (Figure 1-b, Section 2.1.2). This provides high expressiveness while maintaining a small parameter count.

#### 2.1.1 Multiple Size Path Parallelization
The novelty of the Inception block lies in (1) branching the block into parallel paths of different depths to aim for improved expressiveness per block (Figure 1-a). It combines feature maps in the channel direction after parallel implementation of 4 paths: [[1 x 1], [3 x 3], [5 x 5] convolution layers + max pooling layer] (Figure 1-a).

This "multiple size path parallelization (=widening)" allows each Inception module to express the synthesis of features from various convolution sizes.

#### 2.1.2 Dimension Reduction for Lightweight Design
However, with the 4-parallel path configuration of the Inception block (Figure 1-a), the parameter count becomes too large when connecting multiple blocks, making it difficult to train a deep network (with the learning techniques of that time).

Therefore, they introduced the 1x1 convolution proposed in NiN (Network-in-Network) [Lin et al., 2014] at the beginning of each of the three paths to perform feature dimension reduction (in the channel direction) at each spatial position, performing convolution layers after reducing channel numbers (Figure 1-b). This form of block was used in Inception v1.

This achieved computational efficiency for the Inception module. Inception v1 achieved comparable accuracy with significantly fewer parameters than the contemporary VGGNet.

*Note: In ILSVRC2014, Inception v1 took first place with an error rate of 6.6%, while VGGNet took second place with an error rate of 7.3%.*

### 2.2 Global Average Pooling
The "global average pooling" proposed in Network-in-Network (NIN) [Lin et al., 2014] as a replacement for fully connected layers in the final head was also used in Inception v1.

This greatly reduced the CNN's parameter count as the "layer-to-layer weight parameters" used in the fully connected layers of the classifier head became unnecessary when replaced with global average pooling.

### 2.3 Auxiliary Classifiers: Stabilizing Backpropagation to Early Layers
When introducing a total of 22 convolution layers in depth, there was a problem with gradients not propagating well to early layers during backpropagation. Therefore, InceptionNet v1 proposed using auxiliary classifiers as learning support for stable training.

It adds classifier branches (composed of fully connected layers similar to AlexNet's final layers) as heads twice in the middle layers of Inception. Then, it takes additional cross-entropy loss at the end output of the added auxiliary classifier heads and adds it to the total loss.

This allowed easier backpropagation of loss to intermediate parts, enabling larger gradients to be added to backpropagation around the middle area. Thanks to this, InceptionNet could avoid gradient vanishing and propagate gradients to early layers even with its deep large-scale structure, enabling stable learning convergence (auxiliary classifier branches are used only for training, not for testing).

This idea of adding auxiliary classifiers to intermediate layers and output layers to stabilize deep network learning was later applied in ACGAN (Auxiliary Classifier GAN) [Odena et al., 2017] and others, becoming a popular approach in the realm of deep generative models for image generation and image-to-image translation.

## 3. v2: Introduction of Batch Normalization
The batch normalization paper [Ioffe and Szegedy, 2015] was published by the same Google authors as InceptionNet.

Inception v2, which introduced batch normalization to Inception, was experimented with in the batch normalization paper, and this is called version 2 of InceptionNet. In later research, it is often called BN-Inception.

## 4. v3: Improvement of Inception Blocks
[Szegedy et al., 2016] proposed Inception v3 as an improved version reconsidering Inception v1.

In Inception v3, both image recognition performance and model efficiency were improved by introducing "new Inception blocks utilizing factorization (Section 4.1)" and "efficient spatial size reduction blocks (Section 4.2)." This evolved it into a "network structure with better balance between width and depth" compared to Inception v1.

The network configuration of InceptionNet v3 (excluding auxiliary classifier branches) is as follows:

- Early convolution + pooling layers
- Inception block A (Figure 2-a)
- Spatial size reduction block (Figure 4-b)
- Inception block B (Figure 2-b)
- Spatial size reduction block (Figure 4-b)
- Inception block C (Figure 3)
- Classifier head layers

Additionally, it makes error backpropagation easier in deep networks by branching the auxiliary classifier branch from the second spatial size reduction block.

Furthermore, label smoothing (Section 4.3) was proposed as a regularization term for the loss function, which would later be widely used in deep models.

### 4.1 Parameter Reduction through Factorization
Inception v3 uses module A, module B (Figure 2), and module C (Figure 3) in sequence, taking as input the feature map [35 x 35 x 288] first convolved with six 3x3 convolution layers + 3x3 pooling layer.

All three modules utilize "factorization of convolution layers." This achieved both (1) parameter reduction and (2) performance improvement for InceptionNet v3 as a whole.

First, "module A (Figure 2-a)" was proposed, which factorizes the [5 x 5] convolution path used in Inception v1's Inception block (Figure 1) into two layers of [3 x 3] convolutions for parameter reduction (same concept as VGGNet). Channel-direction dimension reduction is performed with [1 x 1] convolution layers before each convolution, which is the same technique as Inception v1 (Figure 1-b).

Also, "module B (Figure 2-b)" using asymmetric factorization was proposed. It reduces computational cost by factorizing the original [n x n] convolution into two layers of [n x 1] and [1 x n] with equivalent receptive fields.

"Module C (Figure 3)" was also proposed, which promotes high-dimensional features by branching paths at the module's end. High-dimensionalization promotes more disentangled (convolution) features in each path (principle 2 in the paper section 2). This also improves convergence and speeds up learning.

### 4.2 Effective Spatial Size Reduction
Inception v3 proposed using a new "spatial size reduction block (Figure 4-b blue frame)" to perform spatial size reduction efficiently.

While Inception v1 uses 1x1 convolutions within blocks to reduce feature dimensions in the channel direction at each spatial position, there were still issues with spatial size reduction, with two areas for improvement:

Type 1 (Figure 4-a left):
When pooling before the Inception block
The middle feature map becomes a bottleneck, hindering improvement in expressiveness.

Type 2 (Figure 4-a right):
When placing the Inception block first
Computational cost triples.

Therefore, Inception v3 proposed and used an "improved spatial size reduction block" (Figure 4-b blue frame). Using this improved version allows increasing filter numbers while reducing spatial size. This enabled spatial size reduction while maintaining low computational cost and avoiding bottleneck representations.

### 4.3 Label Smoothing
The 2015-2016 research of Inception v3 (and ResNet) was motivated by the desire to increase layers and make networks deeper than the contemporary state-of-the-art CNN backbone VGGNet (16/19 layers). However, increasing layers further increased the number of model parameters, raising the risk of CNN overfitting.

As a countermeasure, Inception v3 proposed label smoothing as a regularization method for the softmax loss function (Figure 5). Label smoothing achieves regularization by creating smoothed pseudo-labels that add a small noise distribution to all classes in the distribution created by Softmax from the correct label, and training with these pseudo-labels.

## 5. Inception-v4 and Inception-ResNet