# Batch Normalization Explained

Batch Normalization (BN) is a technique used in deep neural networks to stabilize and accelerate training by normalizing the inputs to layers within the network.

## The Core Idea 💡

Normalize the activations coming out of a layer *for each feature (channel) independently* within the **current mini-batch** so they have **zero mean** and **unit variance**.

## How it Works (During Training) ⚙️

For a mini-batch $B = \{x_1, ..., x_m\}$ and considering a single activation feature:

1.  **Calculate Mini-Batch Mean:**
    $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$
2.  **Calculate Mini-Batch Variance:**
    $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$
3.  **Normalize:** Using the mini-batch mean and variance (adding a small $\epsilon$ for numerical stability):
    $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
4.  **Scale and Shift:** Introduce two learnable parameters, $\gamma$ (gamma) for scaling and $\beta$ (beta) for shifting:
    $y_i = \gamma \hat{x}_i + \beta$

*   The parameters $\gamma$ and $\beta$ are learned during backpropagation alongside the network's weights.
*   This process is applied independently to each feature/channel dimension.

## Why Scale and Shift ($\gamma$ and $\beta$)? 🤔

If we just normalized to zero mean and unit variance, the network's ability to represent information might be limited. Forcing activations into this specific distribution might not always be optimal for the subsequent layers or activation functions.

*   $\gamma$ and $\beta$ give the network **flexibility**.
*   They allow the network to learn the **optimal scale and shift** for the normalized activations.
*   If needed, the network can even learn parameters ($\gamma = \sqrt{\sigma_B^2 + \epsilon}, \beta = \mu_B$) that effectively **undo the normalization**, restoring the original activation distribution if that proves beneficial.

## Batch Normalization During Inference 🧐

During inference (when making predictions), we often process samples one by one, so calculating mini-batch statistics isn't feasible or representative.

*   Instead, BN layers maintain **running averages** of the mean ($\mu_{pop}$) and variance ($\sigma^2_{pop}$) encountered across *all* mini-batches during training.
    *   These are typically updated using a momentum term:
        *   $\mu_{pop} = \alpha * \mu_{pop} + (1 - \alpha) \times \mu_B$
        *   $\sigma_{pop}^2 = \alpha * \sigma_{pop}^2 + (1 - \alpha) \times \sigma_B^2$
        * where $\alpha$ is a momentum parameter.
*   At inference time, these fixed *population* statistics are used for normalization:
    $ \hat{x} = \frac{x - \mu_{pop}}{\sqrt{\sigma_{pop}^2 + \epsilon}} $
*   The learned $\gamma$ and $\beta$ parameters from training are still applied:
    $$ y = \gamma \hat{x} + \beta $$

## Placement 📍

*   It's common practice to place the Batch Norm layer **after** the Convolutional or Fully Connected layer and **before** the non-linear activation function (like ReLU).
    *   `Conv / FC -> Batch Norm -> Activation (ReLU)`
*   However, variations in placement exist in different architectures.

*(Note: While originally thought to primarily combat "internal covariate shift", recent research suggests BN's effectiveness might be more related to smoothing the optimization landscape.)*