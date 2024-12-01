# Set up


![](https://lh6.googleusercontent.com/proxy/GZzRcuxWfjywsYP0tHbDXG9sd8M0X85QSE6cJ7XGnu3S1D63w81USLd3eCwjF_pNjvZko5fwCfRSoEE_f1G1sTgP8pCccUk9FI992V29BucEeeMiU_LvrA)


In this course, we will use **Python** - a powerful and beginner-friendly programming language that's perfect for data science and AI. Python has become the go-to language for machine learning thanks to its:

- Simple, readable syntax that's great for learning
- Rich ecosystem of AI/ML libraries like TensorFlow and PyTorch
- Huge community creating tools and tutorials
- Free and open-source nature
- Excellent data analysis and visualization capabilities

Whether you're new to programming or experienced with other languages, Python will give us everything we need to explore applied soft computing.

For most parts, we will use [Google Colab](https://colab.research.google.com/) to run a Jupyter notebook.
This course involves a lot of coding and data analysis. For most parts, we will use [Google Colab](https://colab.research.google.com/) to run a Jupyter notebook.


# Trouble shooting

## Cannot plot a graph with igraph on Google Colab

Google Colab has many packages pre-installed. However, they do not include some pacages for network analysis like `igraph` and `graph-tool`.

**Installing igraph**
Create a cell on top of the notebook and run the following code to install the igraph.
```
!sudo apt install libcairo2-dev pkg-config python3-dev
!pip install pycairo cairocffi
!pip install igraph
```

**Installing graph-tool**
Create a cell on top of the notebook and run the following code to install the graph-tool.
```
!wget https://downloads.skewed.de/skewed-keyring/skewed-keyring_1.0_all_$(lsb_release -s -c).deb
!dpkg -i skewed-keyring_1.0_all_$(lsb_release -s -c).deb
!echo "deb [signed-by=/usr/share/keyrings/skewed-keyring.gpg] https://downloads.skewed.de/apt $(lsb_release -s -c) main" > /etc/apt/sources.list.d/skewed.list
!apt-get update
!apt-get install python3-graph-tool python3-matplotlib python3-cairo

# Colab uses a Python install that deviates from the system's! Bad colab! We need some workarounds.
!apt purge python3-cairo
!apt install libcairo2-dev pkg-config python3-dev
!pip install --force-reinstall pycairo
!pip install zstandar
````
