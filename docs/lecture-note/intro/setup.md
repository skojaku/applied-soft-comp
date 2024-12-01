
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
