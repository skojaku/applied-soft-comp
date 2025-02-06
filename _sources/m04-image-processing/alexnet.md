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

# AlexNet: A Breakthrough in Deep Learning

## The ImageNet Challenge and the Deep Learning Revolution

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) represented one of the most ambitious and challenging tasks in computer vision: classifying images across 1000 different categories. Before 2012, the best performing systems struggled with error rates above 25%, relying primarily on hand-crafted features and traditional machine learning approaches. The sheer scale of the dataset—--with over 1.2 million training images—--posed significant computational challenges that many believed would make deep learning approaches impractical.

```{figure} https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs11263-015-0816-y/MediaObjects/11263_2015_816_Fig2_HTML.gif
The ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
```

```{tip}
Prior to AlexNet, the dominant approaches in computer vision relied heavily on hand-engineered features like SIFT (Scale-Invariant Feature Transform) and HOG (Histogram of Oriented Gradients). These methods required significant domain expertise and often failed to generalize well across different types of images.
```

In 2012, Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton submitted a deep convolutional neural network that achieved a top-5 error rate of 16.4%---an unprecedented improvement of over 10 percentage points compared to the second-best entry {footcite}`krizhevsky2012alexnet`.
This represents a paradigm shift in computer vision and machine learning towards deep learning.

```{figure} https://viso.ai/wp-content/uploads/2024/02/imagenet-winners-by-year.jpg
:width: 100%
:align: center

Top 5 error rates of the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) from 2010 to 2017. AlexNet reduced the error rate by over 10 percentage points compared to the best performing method based on human-crafted features in 2011.
```


## Key Innovations of AlexNet

The idea of using multiple layers of neurons has been around for a long time. However, deep neural networks have cricial issues in training. AlexNet introduced several crucial innovations to address this challenge, which have since become standard practices in deep learning:

### ReLU Activation Function

```{figure} https://miro.medium.com/v2/resize:fit:474/1*HGctgaVdv9rEHIVvLYONdQ.jpeg
:width: 100%
:align: center

Sigmoid and ReLU activation functions. Sigmoid is prone to the vanishing gradient problem due to the plateau for input $x$ far from zero. ReLU, on the other hand, does not suffer from this problem as long as $x>0$. The image is taken from https://medium.com/@jarajan123/a-comparative-analysis-relu-vs-sigmoid-activation-functions-fa1dbe481d80
```

Traditional neural networks with multiple layers often suffer from *the vanishing gradient problem*. It is a phenomenon where the gradient of parameters in early layers approaches zero, hindering the training process. This is attributed to the activation functions, the most common one being the sigmoid function.

The sigmoid function is defined as:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

The sigmoid function has a gradient of:

$$
f'(x) = f(x) (1 - f(x))
$$

Input $x$ is random and effectively distributed around zero when the neural network is initialized. The gradient is thus large. However, as the neural network learns the data, the input signal $x$ tends to be far from zero, making the gradient of the sigmoid function approach zero.

The Rectified Linear Unit (ReLU) was proposed in the previous year of AlexNet by one of the authors of AlexNet, Hinton {footcite}`nair2010rectified`.

The Rectified Linear Unit (ReLU) is defined by a simple operation:

$$
f(x) = \max(0, x)
$$

The gradient of the ReLU function is:

$$
f'(x) = \begin{cases}
0 & \text{if } x \leq 0 \\
1 & \text{if } x > 0
\end{cases}
$$

The ReLU function solves the vanishing gradient problem since its gradient is always either 0 or 1 when the input is positive. Additionally, ReLU is computationally efficient because it only needs to check if the input is greater than zero, making it faster than more complex activation functions like sigmoid.

```{note}
ReLU suffers from so-called "dying neurons" problem, where neurons can get stuck in the dead state (i.e., $x \leq 0$) and never activate, since the gradient is zero. This issue leads to various activation functions that add a small positive value for $x \leq 0$ to the ReLU function. For example, Leaky ReLU and Parametric ReLU (PReLU) are two such activation functions defined as:

$$
f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$

where $\alpha$ is a small positive constant.

```{figure} https://media.licdn.com/dms/image/v2/D4D12AQH2F3GJ9wen_Q/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1688885174323?e=2147483647&v=beta&t=dY_S6xeNsRCIvpIrjrPFzq8qgHPgmP4e_HLaA15ufPM
:width: 100%
:align: center

Comparison of various activation functions. The image is taken from https://www.linkedin.com/pulse/activation-functions-heba-al-haddad
```

### Dropout Regularization

Another key limitation of deep neural networks is overfitting. Overfitting occurs when the neural network learns the training data too well, resulting in poor generalization to unseen data. It is thus crucial to prevent overfitting by "regularizing" the neural network.

A traditional way to prevent overfitting is to use L2 regularization, which adds a penalty term to the loss function to prevent the weights from becoming too large. While this method is effective, an easier alternative is *dropout* {footcite}`srivastava2014dropout`.

```{figure} https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/04/1IrdJ5PghD9YoOyVAQ73MJw.gif
:width: 100%
:align: center

Dropout in action. The image is taken from https://primo.ai/index.php?title=Dropout
```

Dropout is a regularization technique that randomly drops out neurons during training to prevent overfitting. It works in two modes:

1. During training, each neuron has a probability $p$ (typically 0.5) of being temporarily "dropped out" of the network. Specifically, for a given input $x$, the output $y$ of the neuron is given by
  $$
  y = r \cdot x, \quad \text{where } r \sim \text{Bernoulli}(p)
  $$
2. During inference, all neurons are used without dropping out. However, because all neurons can attend to the same input in the subsequent layers, their outputs might be changed. To compensate for this, the outputs of the neurons are scaled by the "keep probability" $1-p$ to maintain the distribution of the outputs.
  $$
  y = (1-p) x
  $$

This technique effectively forces the network to learn redundant representations, as it cannot rely on any single neuron being present. During inference, all neurons are used, but their outputs are scaled by the dropout probability to maintain consistent expected values.

```{note}
An alternative way to implement dropout is so-called *inverse dropout*, where, instead of scaling the input $x$, scale the weights of the neurons by $1/(1-p)$ **during training**. During inference, no scaling is applied to both the input and the weights. This is what is used in TensorFlow.
```


### Local Response Normalization (LRN)

Similar to how our eyes adjust to see details in both bright and dark areas of an image (like being able to see both the details of a person standing in the shadows and the bright sky behind them), Local Response Normalization (LRN) helps neural networks balance and normalize feature responses across channels. LRN is a normalization technique for CNNs originally proposed in AlexNet {footcite}`krizhevsky2012alexnet` that normalizes feature map responses across adjacent channels to enhance local contrast.

The LRN operation for a given activity $a^i_{x,y}$ at position $(x,y)$ in the $i$-th channel of the feature map was defined as:

$$
b^i_{x,y} = a^i_{x,y} / \left(k + \alpha \sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)} (a^j_{x,y})^2\right)^\beta
$$

where the constants $k$, $n$, $\alpha$, and $\beta$ were determined through validation. Note that this is the channel-wise normalization. See the original paper for more details.

LRN normalizes each neuron's response based on the activity of its neighboring neurons across the channels. It may improve CNN generalization by making features locally contrast-invariant and potentially stabilizing training. However, LRN has limitations, one of which is that it can only be applied to multi-channel visual features and might potentially reduce discriminative power by altering the overall representation.
A remedy for this is Local Contrast Normalization (LCN) {footcite}`ortiz2020local`, which normalizes within a local window of a single channel.

## Architecture and Implementation

AlexNet's architecture consisted of five convolutional layers followed by three fully connected layers.

```{figure} ../figs/alexnet-architecture.jpg
:width: 50%
:align: center

Detailed architecture diagram showing the parallel GPU implementation
```

The network was split across two GPUs due to memory constraints of the hardware available at the time:

* **Input image**: [224 x 224] normalized, 3-channel color image (with color whitening, section 3.2.1)
* **Conv1**: Convolutional layer - [11 x 11] kernel x 96 channels, stride = 4, padding = 2
* **Activation function**: ReLU + Local Response Normalization (section 3.2.2)
* **P1**: Pooling layer - Overlapping max pooling, [3 x 3] kernel, stride = 2
* **Conv2**: Convolutional layer - [5 x 5] kernel x 256 channels, stride = 1, padding = 2
* **Activation function**: ReLU + Local Response Normalization (section 3.2.2)
* **P2**: Pooling layer - Overlapping max pooling, [3 x 3] kernel, stride = 2
* **Conv3**: Convolutional layer - [3 x 3] kernel x 384 channels, stride = 1, padding = 1
* **Conv4**: Convolutional layer - [3 x 3] kernel x 384 channels, stride = 1, padding = 1
* **Conv5**: Convolutional layer - [3 x 3] kernel x 256 channels, stride = 1, padding = 1
* **P3**: Pooling layer - Overlapping max pooling, [3 x 3] kernel, stride = 2
* (During training only: **Dropout**)
* **FC6**: Fully connected layer - 9216 (= 256 x 6 x 6) → 4096
* **Activation function**: ReLU
* (During training only: **Dropout**)
* **FC7**: Fully connected layer - 4096 → 4096
* **Activation function**: ReLU
* **FC8**: Fully connected layer - 4096 → 1000
* **Output**: 1000-dimensional vector probability distribution (output probability for each dimension) using softmax function

It appears to be complicated. However, the architecture follows a clear pattern common to many convolutional neural networks.
1. It starts with convolutional layers that extract features from the input image, followed by pooling layers that reduce spatial dimensions while preserving important information.
2. The network begins with larger kernels (11x11, 5x5) to capture broad features and transitions to smaller ones (3x3) for more detailed features.
3. The three consecutive 3x3 convolutions (Conv3, Conv4, Conv5) effectively create a larger receptive field while using fewer parameters than a single large kernel.
4. The network then flattens the spatial features and passes them through fully connected layers, progressively reducing dimensions from 9216 to 4096 and finally to 1000 classes.
5. The use of ReLU activation functions helps prevent vanishing gradients, while Local Response Normalization and Dropout serve as regularization techniques to prevent overfitting.

AlexNet consists of two paths that are parallelly executed on two GPUs. This is to reduce the memory footprint of the network, since at that time, the memory of a single GPU was insufficient for the network (i.e., 3GB of memory for a single GPU).

- The first Convolutional layer (Conv1) produces a feature map of 96 channels, which are split into two halves, where 48 channels for each GPU. The pooling and normalization are applied to each half.
- The second Convolutional layer (Conv2) independently processes the feature maps of the two GPUs, producing two 128-channel feature maps.
- In the third Convolutional layer (Conv3), the feature maps of the two GPUs interact with each other, producing two 192-channel feature maps, each of which lives on a different GPU.
- The fourth and fifth Convolutional layer (Conv4 and Conv5) independently processes the feature maps of the two GPUs.

```{tip}
The use of GPUs for training neural networks was not new, but AlexNet demonstrated their practical necessity for training large-scale networks on real-world datasets. This helped establish GPU computing as a cornerstone of modern deep learning.
```

## Implementation

Interested in implementing AlexNet? You can find the implementation in
[Writing AlexNet from Scratch in PyTorch | DigitalOcean](https://www.digitalocean.com/community/tutorials/alexnet-pytorch)
