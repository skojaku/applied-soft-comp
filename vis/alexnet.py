# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "markdown==3.7",
#     "matplotlib==3.10.1",
#     "numpy==2.2.3",
#     "pandas==2.2.3",
#     "pillow==11.1.0",
#     "requests==2.32.3",
#     "scikit-learn==1.6.1",
#     "seaborn==0.13.2",
#     "torch==2.6.0",
#     "torchvision==0.21.0",
#     "tqdm==4.67.1",
#     "transformers==4.49.0",
# ]
# ///

import marimo

__generated_with = "0.12.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        <div style="display: flex; align-items: center;">
            <div style="flex: 1;">
                <span style="font-size: 1.5em;">AlexNet: Hands on</span>
                <p>Sadamori Kojaku</p>
            </div>
            <div style="flex: 1;">
                <img src="https://i.ytimg.com/vi/ZUc0Mib5DeI/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLAExZC6AMn-c7XTAeA2k30TJ16MgQ" width="400">
            </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        /// tip | How to run this notebook

        To run the notebook, first download it as a `.py` file, then use the following steps:

        Install **marimo**:
        ```bash
        pip install marimo
        ```

        Install **uv** (a Python package manager that automatically manages dependencies):
        ```bash
        pip install uv
        ```

        Launch the notebook
        ```bash
        marimo edit --sandbox <filename>.py
        ```

        The notebook will open in your web browser. All necessary packages will be installed automatically in a dedicated virtual environment managed by **uv**.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _():
    # Load AlexNet model
    return


@app.cell
def _():
    import torchvision.models as models

    #alexnet = models.alexnet(pretrained=True)
    alexnet = models.vgg16(pretrained=True)
    #alexnet = models.inception_v3(pretrained=True)
    alexnet.eval()
    print(alexnet.eval())
    return alexnet, models


@app.cell(hide_code=True)
def _():
    # Image Data
    return


@app.cell
def _():
    from PIL import Image
    import requests
    from io import BytesIO

    # Download and preprocess an example image
    url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"

    # Real Bearcat
    url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSnzXi8Bfi2lq22uqFW3zF0bt8Ir7DjmjTJtg&s"

    # Bearcat
    url = "https://blog.suny.edu/wp-content/uploads/2014/03/Binghamton-baxter.jpg"

    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    image
    return BytesIO, Image, image, requests, url


@app.cell(hide_code=True)
def _():
    # Preprocessing (Resizing, cropping, and normalization)
    return


@app.cell
def _(image):
    import torchvision.transforms as transforms

    # Preprocessing pipeline for AlexNet
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),  # Resize shorter side to 256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),  # Convert to tensor [C, H, W], scale to [0, 1]
            transforms.Normalize(  # Normalize using ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Preprocess the input image
    image_data = preprocess(image)
    return image_data, preprocess, transforms


@app.cell
def _(image_data):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(figsize=(15, 5), ncols=3)

    axes[0].imshow(image_data[0])
    axes[1].imshow(image_data[1])
    axes[2].imshow(image_data[2])
    return axes, fig, plt


@app.cell(hide_code=True)
def _():
    # Inference
    return


@app.cell
def _(alexnet, image_data):
    import torch

    input_tensor = image_data.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = alexnet(input_tensor)

    print("Size:", output.shape)
    print(output[0][:10])
    return input_tensor, output, torch


@app.cell
def _(output, torch):
    # Get top-5 prediction
    logits, predicted = torch.topk(output[0], 5)

    print(predicted)
    return logits, predicted


@app.cell
def _(logits, plt, predicted):
    from torchvision.models import AlexNet_Weights
    import seaborn as sns

    weights = AlexNet_Weights.DEFAULT
    class_labels = weights.meta["categories"]

    # Prepare data
    labels = [class_labels[i] for i in predicted]
    scores = logits

    # Plot
    sns.barplot(x=scores, y=labels)
    plt.xlabel("Probability")
    plt.title("Top-5 AlexNet Predictions")
    plt.show()
    return AlexNet_Weights, class_labels, labels, scores, sns, weights


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import inspect
    import markdown
    return inspect, markdown, mo


if __name__ == "__main__":
    app.run()
