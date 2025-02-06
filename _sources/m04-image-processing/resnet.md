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

# ResNet (Residual Neural Networks)

*Why did simply adding more layers to CNNs (like VGGNet or InceptionNet) fail to yield the expected performance gains—and sometimes even degraded accuracy?*

Residual Neural Networks (ResNet) fundamentally changed the landscape of deep CNN training by introducing **residual connections** (a.k.a. skip connections). By stacking a series of **residual blocks**, ResNet enabled training CNNs with dozens or even hundreds of layers without succumbing to the **vanishing gradient problem**. Today, ResNet is considered one of the most important innovations in the history of deep learning, influencing architectures like **ResNeXt** and even **Transformers**.

```{note}
ResNeXt is an improvement over ResNet proposed by the same research group {footcite}`xie2017aggregated`. It widens the residual blocks via grouped convolutions, achieving higher performance without drastically increasing depth.
```

## Introduction and Context

ResNet was introduced in {footcite}`he2016deep` to address a key challenge at the time: **CNNs deeper than about 20 layers were difficult to optimize and often performed worse than shallower counterparts**. Despite the success of VGGNet (16 or 19 layers) and InceptionNet, researchers still faced two major issues when pushing CNNs to 50 layers or more:

1. **Degradation Problem**: Simply stacking more layers often *degraded* accuracy, rather than improving it.
2. **Long Training Times**: Extremely deep CNNs took a long time to converge, especially if the network was prone to vanishing or exploding gradients.

The ResNet solution was surprisingly simple yet groundbreaking: add skip connections that carry the original inputs across a few layers unmodified, letting the network focus on modeling the **residual**.

## ResNet in Detail

### Why Going Deeper Was Difficult

*Shouldn't deeper networks always perform better because they have more parameters and expressive power?*

In theory, **deeper** CNNs can capture richer, more complex patterns. However, two issues hindered progress:

1. **Degradation Problem**
   Even with techniques like batch normalization, adding more layers beyond ~20 caused training error to *increase*, not decrease. This phenomenon was *not* simply due to overfitting—rather, the deeper network failed to optimize properly.

2. **Longer Training and Vanishing Gradients**
   As more layers are added, gradients can vanish (or explode). Backprop had trouble sending meaningful error signals all the way to early layers, causing them to learn slowly or not at all.

### Key Proposal: Residual Learning with Skip Connections

*What if each stack of layers simply learned a correction (residual) to the identity mapping?*

A **residual block** consists of two (or three) convolutions grouped together, plus a **skip connection**:

- **Residual Path**: A few convolution layers (for example, two 3×3 conv layers) modeling a function $ F(\mathbf{x}) $.
- **Skip (Identity) Path**: A direct path for $\mathbf{x}$ to bypass the convolutions entirely.

At the end of the block, the skip path is added elementwise to the residual path:
$$
\mathbf{y} = F(\mathbf{x}) + \mathbf{x}.
$$

In PyTorch, you can implement a basic residual block as follows:

```{code-cell} ipython3
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out
```

```{figure} https://www.researchgate.net/publication/364330795/figure/fig7/AS:11431281176036099@1689999593116/Basic-residual-block-of-ResNet.png
:name: resnet-block
:align: center
:width: 50%
A basic 2-layer residual block (left) vs. a plain block without skip (right). The skip connection allows the input $\mathbf{x}$ to directly add to the block’s output.
```

By stacking many such blocks, the network effectively *cascades* small residual changes across layers. The key benefits are:

1. **Easier Optimization**
   Instead of learning a full mapping $\mathbf{y} = G(\mathbf{x})$, the block learns only the difference $G(\mathbf{x}) - \mathbf{x}$. This decomposition often proves easier to optimize.

   ```{note}
   If the optimal mapping is close to identity (i.e., the layer isn't very important), the network can easily "skip" it by learning $F(\mathbf{x}) \approx 0$. If a more complex transformation is needed, the residual path can still learn it. This makes training more robust—the network doesn’t have to work as hard to preserve important information through deep layers.
   ```

2. **Ensemble-Like Behavior**
   When you chain $N$ residual blocks, you effectively create numerous paths for gradient flow—some skip many layers, some pass through multiple convolutions. This variety of gradient routes can speed convergence and reduce the risk of vanishing gradients {footcite}`veit2016residual`.

    ```{figure} https://arxiv.org/html/2405.01725v1/x28.png
    :name: resnet-gradient-flow
    :align: center
    :width: 100%
    The gradient flow in ResNet with skip connections.
    ```

3. **Deeper Without Degradation**
   ResNet-50, -101, and -152 can be trained without suffering the performance drop typical of overly deep “plain” networks.

### Bottleneck Blocks for Deep ResNet

ResNet has some variants depending on the depth. For deep ResNet, the **bottleneck** design is used to maintain computational efficiency.

```{figure} https://i.sstatic.net/kbiIG.png
:name: resnet-bottleneck-block
:align: center
:width: 80%
A bottleneck block of ResNet.
```

This bottleneck block consists of three convolutions instead of two, where:
- the first $1 \times 1$ conv **reduces** the feature dimension.
- the second $3 \times 3$ conv operates on this reduced dimension.
- the third $1 \times 1$ conv **restores** the dimension.

This approach shrinks the intermediate feature map, saving computational cost while retaining overall representational capacity. It was inspired by InceptionNet’s “bottleneck” idea {footcite}`szegedy2016inception,szegedy2015going`.

```{tip}
**ResNet-50**, **ResNet-101**, and **ResNet-152** all use bottleneck blocks. While they have more layers, they remain computationally feasible and yield progressively better accuracy on ImageNet.
```

## ResNeXt: A ResNet Improvement

*What if we can widen the residual blocks without drastically increasing overall parameters?*

**ResNeXt** {footcite}`xie2017aggregated` is an evolution of ResNet that:
1. **Splits** the bottleneck conv pathway into multiple “cardinality” groups (e.g., 32 groups).
2. **Aggregates** those parallel paths (grouped convolutions) back into a single output.

By increasing **cardinality** (the number of parallel conv groups) instead of just adding more channels or layers, ResNeXt achieves better accuracy with moderate complexity. This approach also draws on the idea of Inception’s multi-branch parallel conv, but unifies them into a single grouped-convolution block.

```{figure} https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_4.32.52_PM.png
:name: resnext-block
:align: center
:width: 50%
A basic block of ResNeXt, showing multiple grouped-conv “paths” that are aggregated.
```

## Implementation of ResNet

- [Writing ResNet from Scratch in PyTorch](https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch)

## Summary

1. **Residual Learning**
   ResNet overcame the degradation problem by framing deeper CNNs as a series of residual blocks, each learning a function $ F(\mathbf{x}) $ that is added to $\mathbf{x}$.

2. **Scalability**
   With skip connections, ResNet-50, -101, and -152 exhibit higher accuracy without the optimization collapse typical of deeper plain networks.

3. **Bottleneck & Beyond**
   For high-depth architectures, the bottleneck design $(1\times1 \to 3\times3 \to 1\times1)$ improves efficiency. ResNeXt further extends ResNet by widening these pathways via grouped convolutions.

4. **Lasting Impact**
   Residual connections are now ubiquitous—not just in CNNs but also in Transformers, large-scale language models, U-Nets, and many other architectures. They simplify optimization and significantly improve gradient flow in very deep models.

```{note}
ResNet’s simplicity made it a foundation for many follow-up architectures. Unlike designs with complex branching (e.g., Inception blocks), ResNet remains easy to implement, debug, and extend—an important factor behind its widespread adoption.
```

## Suggested Exercises

1. **Implement a Basic (Non-Bottleneck) Residual Block**
   - Create a two-convolution block with skip connections.
   - Test it on random data to confirm dimensions match.
2. **Train a Small ResNet**
   - Implement ResNet-18 or ResNet-34 from scratch on a smaller dataset (e.g., CIFAR-10).
   - Observe the training curve and compare to a plain CNN of the same depth.
3. **Experiment with Bottleneck Blocks**
   - Convert your ResNet-34 to a bottleneck-based ResNet-50-like structure.
   - Check the parameter count and performance difference on CIFAR-10 or a subset of ImageNet.

```{footbibliography} references.bib
:style: unsrt
:filter: docname in docnames
```