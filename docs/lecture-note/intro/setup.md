# Set up

## Tools

![](https://lh6.googleusercontent.com/proxy/GZzRcuxWfjywsYP0tHbDXG9sd8M0X85QSE6cJ7XGnu3S1D63w81USLd3eCwjF_pNjvZko5fwCfRSoEE_f1G1sTgP8pCccUk9FI992V29BucEeeMiU_LvrA)

In this course, we will use **Python** - a versatile and easy-to-use high-level programming language widely adopted by scientists across disciplines. Python is an excellent choice for scientific computing and data analysis for several key reasons:

- **Low barrier to entry**: Unlike lower-level languages like C/C++/Fortran, Python doesn't require specialized programming skills, allowing you to focus more on science than coding details.

- **Rich ecosystem**: Python's extensive scientific libraries and tools make it easy for newcomers to leverage existing code. The ability to quickly create and share libraries means new functionality becomes available rapidly.

- **Cross-platform compatibility**: Thanks to the Python interpreter, code written in Python can run across different platforms and architectures with minimal modification.

- **Interactive development**: The Python interpreter, especially when used with Jupyter notebooks, enables interactive code development and experimentation. This makes it easy to explore data, test ideas, and document your work in a reproducible way.

For this course, we'll be using Python 3, and we recommend working with Jupyter notebooks which combine code, documentation, and results (although personally, I avoid notebooks as much as possible in research for maintainability and reusability).


[How Did Python Become a Data Science PowerHouse (PyData Seattle 2017)](https://www.youtube.com/watch?v=fk8ATuMUltU)


For those unfamiliar with programming in Python, there are several online tutorials available that can provide a basic introduction to the language ([codecademy](https://www.codecademy.com/learn/learn-python), [learnpython](https://www.learnpython.org/)).


- All materials, including exercise coding materials, pen-and-paper worksheet, slides, and lecture notes in this course are located in the official course GitHub page: (https://github.com/skojaku/applied-soft-comp)

- Assignemnts will be distributed, collected and graded through GitHub Classroom, an educational infrastructure for coding assignments. The instructor will distribute the links to the assignment to the students, and students will submit their answers through GitHub Classroom. Here is how to submit your assignment: (https://youtu.be/YEp942EbggQ)


## Trouble shooting

### Cannot plot a graph with igraph on Google Colab

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
