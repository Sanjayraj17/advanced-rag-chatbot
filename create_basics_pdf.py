"""
Script to create python_data_science_basics.pdf with clean, fully
separated pages per topic using PyMuPDF (fitz) - no external reportlab required.
"""
import os
import pymupdf

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "python_data_science_basics.pdf")

doc = pymupdf.open()

PAGES_CONTENT = [
    # PAGE 1: PYTHON
    """Python Programming Language

What is Python?
Python is an easy to learn, powerful, and versatile programming language. It features efficient high-level data structures and a simple but effective approach to object-oriented programming. Python's elegant syntax and dynamic typing, together with its interpreted nature, make it ideal for scripting, automation, web development, data science, and rapid application development across multiple domains.

Key Advantages and Features of Python:
- Easy to Learn and Read: Python's syntax is clean and close to plain English, making it one of the most readable and beginner-friendly programming languages.
- Interpreted Language: Python executes code line-by-line, which makes interactive development, rapid prototyping, and debugging fast and straightforward.
- Versatile and General-Purpose: Python is used across many domains including web development, data science, machine learning, artificial intelligence, automation, and scientific computing.
- Extensive Standard Library: Python includes a comprehensive built-in standard library covering file I/O, networking, regular expressions, mathematics, and data serialization.
- Cross-Platform Compatibility: Python runs seamlessly across Windows, macOS, Linux, and Unix platforms without code modification.
- Dynamically Typed: Variables do not require explicit type declarations; Python infers types at runtime, enabling rapid development.
- Object-Oriented and Functional: Python supports multiple programming paradigms including object-oriented programming, procedural programming, and functional programming.
- Massive Ecosystem and Community: Python boasts the largest package ecosystem through PyPI with hundreds of thousands of third-party libraries and an active global community.
- Free and Open Source: Python is open source and freely distributable for personal and commercial use under an OSI-approved license.

Python Applications:
Python powers popular web frameworks like Django and Flask, leading machine learning toolkits, scientific research tools, system automation scripts, and enterprise software.
""",

    # PAGE 2: NUMPY
    """NumPy — Numerical Python

What is NumPy?
NumPy (Numerical Python) is the fundamental library for numerical computing and scientific calculations in Python. It introduces the powerful N-dimensional array object (ndarray), which serves as the core data structure for high-performance numerical data processing. NumPy is implemented in C and Fortran, allowing mathematical operations to execute at compiled-language speeds.

Key Advantages and Features of NumPy:
- High Performance: Implemented in compiled C, NumPy array operations run significantly faster than standard Python lists, often 10 to 100 times faster.
- Vectorization: Enables element-wise mathematical and logical operations on entire arrays without slow explicit Python loops.
- Memory Efficiency: NumPy arrays store data in contiguous memory blocks with homogeneous data types, dramatically reducing memory overhead compared to Python lists.
- Broadcasting: Automatically aligns and broadcasts operations across arrays of differing shapes and dimensions without requiring manual memory duplication.
- Comprehensive Linear Algebra and Math Operations: Provides built-in functions for matrix multiplication, eigenvalues, singular value decomposition, Fourier transforms, random sampling, and statistics.
- Foundation of the Python Scientific Stack: Core libraries including Pandas, SciPy, and Scikit-learn are built directly on top of NumPy arrays.
- Multi-Dimensional Array Manipulation: Offers powerful slicing, indexing, reshaping, transposing, and stacking methods for multi-dimensional data structures.

NumPy Core Data Structure:
The ndarray is a fixed-size, homogeneous array representing vectors, matrices, or higher-dimensional tensors with uniform data types.
""",

    # PAGE 3: PANDAS
    """Pandas — Data Manipulation and Analysis

What is Pandas?
Pandas is a fast, powerful, and flexible open-source Python library designed specifically for data manipulation, data cleaning, and structured data analysis. Built on top of NumPy, Pandas introduces two primary data structures: the Series (one-dimensional labeled array) and the DataFrame (two-dimensional labeled tabular data structure).

Key Advantages and Features of Pandas:
- Intuitive Tabular Manipulation: Easily inspect, filter, slice, and transform data using rows and columns similar to SQL database tables or spreadsheet workbooks.
- Missing Data Handling: Built-in methods like isna(), dropna(), and fillna() make detecting, filtering, and imputing missing or null data straightforward and robust.
- Flexible File I/O: Effortlessly reads and writes multiple formats including CSV, Excel spreadsheets, JSON files, Parquet, and SQL database queries.
- Powerful GroupBy and Aggregation: The split-apply-combine paradigm enables complex data grouping, grouping transformations, and statistical summaries with minimal code.
- Time Series Support: Comprehensive functionality for date parsing, date range generation, frequency conversion, moving window statistics, and date arithmetic.
- Merging and Joining: Provides database-style merge, join, and concatenate operations to combine diverse datasets across common keys.
- Vectorized String Operations: Built-in string methods enable efficient text cleaning and regex transformations across entire columns simultaneously.

Pandas Core Data Structures:
The Series represents a single labeled column of data. The DataFrame represents a complete table with labeled rows (index) and labeled columns.
""",

    # PAGE 4: MATPLOTLIB
    """Matplotlib — Data Visualization Library

What is Matplotlib?
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It provides an accessible MATLAB-style plotting interface through the pyplot module, alongside an extensive object-oriented API that gives developers fine-grained control over every visual element.

Key Advantages and Features of Matplotlib:
- Complete Customizability: Total control over figure components including axes, tick marks, colors, line widths, legends, annotations, labels, and plot styling.
- Wide Chart Variety: Supports line charts, scatter plots, bar charts, histograms, box plots, heatmaps, pie charts, contour plots, and 3D graphs.
- Direct Integration: Plots data directly from Pandas DataFrames, Series, and NumPy arrays without manual data conversion.
- Multiple Export Formats: Saves publication-quality graphics in formats such as PNG, PDF, SVG, EPS, and JPEG.
- Subplot Layouts: Flexible subplots() and GridSpec interfaces allow complex multi-panel figure layouts in a single canvas.
- Extensible Foundation: Serves as the plotting backbone for high-level data visualization libraries like Seaborn.

Matplotlib Usage:
Commonly imported as 'import matplotlib.pyplot as plt', it offers simple commands like plt.plot(), plt.scatter(), plt.bar(), and plt.show() for rapid visual discovery.
""",

    # PAGE 5: STATISTICAL LEARNING
    """Statistical Learning

What is Statistical Learning?
Statistical learning refers to a vast set of tools and mathematical frameworks for understanding and modeling data. In statistical learning, we model the relationship between a set of input variables (predictors, features, or independent variables denoted as X) and an output variable (response or dependent variable denoted as Y). This relationship is expressed mathematically as Y = f(X) + ε, where f represents the systematic information that X provides about Y, and ε represents a random error term with mean zero.

Why Estimate f?
- Prediction: When inputs X are readily available but output Y is difficult to obtain, we estimate f to predict Y for new observations. Prediction error consists of reducible error (which our estimate minimizes) and irreducible error (inherent variance from ε).
- Inference: Understanding how Y is affected as X changes. Inference helps identify which predictors are most important, the direction and strength of relationships, and whether associations are linear or non-linear.

Parametric vs Non-Parametric Methods:
- Parametric Methods: Assume a specific functional form or shape for f (such as linear regression). This simplifies the problem to estimating a fixed set of coefficients or parameters.
- Non-Parametric Methods: Do not make explicit assumptions about the functional form of f. Instead, they seek an estimate of f that fits the data points closely without being overly rough or wiggly (e.g., K-nearest neighbors, smoothing splines).

Major Categories of Statistical Learning:
- Supervised Learning: Every observation has both predictor measurements X and a corresponding response value Y.
- Unsupervised Learning: For every observation, only predictor measurements X are observed with no associated response variable Y.
""",

    # PAGE 6: SUPERVISED LEARNING
    """Supervised Learning

What is Supervised Learning?
Supervised learning is a fundamental branch of statistical learning and machine learning where an algorithm learns a mapping function from labeled training data. In supervised learning, the dataset consists of input-output pairs (X, Y). The model learns by comparing its predictions against the true known targets Y and updating its parameters to minimize error. Once trained, the model generalizes to make predictions on unseen data.

Two Primary Types of Supervised Learning:
1. Regression: Used when the output response Y is a continuous numerical quantity.
   - Examples: Predicting housing prices, stock valuations, patient blood pressure, or temperature.
   - Common Algorithms: Linear Regression, Ridge Regression, Lasso Regression, Polynomial Regression, Support Vector Regression, and Decision Tree Regressors.
2. Classification: Used when the output response Y is categorical or qualitative (belonging to distinct classes).
   - Examples: Diagnosing disease vs healthy, detecting email spam vs non-spam, or customer churn prediction.
   - Common Algorithms: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Naive Bayes, Decision Trees, and Random Forests.

Key Supervised Learning Concepts:
- Training and Test Sets: Data is split into training data (to fit the model) and test data (to evaluate true generalization performance).
- Overfitting and Underfitting: Overfitting occurs when a model captures random noise in training data and fails on new data; underfitting occurs when a model is too simple to capture underlying patterns.
- Bias-Variance Tradeoff: Balancing model flexibility (low bias) against sensitivity to training data fluctuations (low variance) to minimize test error.
""",

    # PAGE 7: UNSUPERVISED LEARNING
    """Unsupervised Learning

What is Unsupervised Learning?
Unsupervised learning is a branch of statistical learning and machine learning where algorithms discover patterns, structure, and groupings in data without any response labels or target outcomes. In unsupervised learning, the model is provided solely with input features X. Because there are no correct answers or supervisory signals, the goal is to discover meaningful representations, clusters, or associations inherent in the data.

Primary Types of Unsupervised Learning:
1. Clustering: Partitioning observations into distinct groups such that observations within a group are highly similar, while observations in different groups are dissimilar.
   - Common Algorithms: K-Means Clustering, Hierarchical Agglomerative Clustering, and DBSCAN.
   - Applications: Customer market segmentation, document categorization, and image segmentation.
2. Dimensionality Reduction: Compressing high-dimensional feature spaces into a smaller number of informative dimensions while preserving variance and structural geometry.
   - Common Algorithms: Principal Component Analysis (PCA), t-SNE, and UMAP.
   - Applications: Data visualization, noise filtering, and feature preprocessing for supervised models.
3. Association Analysis: Uncovering probabilistic rules and associations between items in transactional data (e.g., Apriori algorithm for market basket analysis).

Challenges of Unsupervised Learning:
Unlike supervised learning, there is no ground truth objective to optimize or directly validate. Evaluation requires domain expertise, heuristic metrics (like silhouette scores), and exploratory visualization.
"""
]

rect = pymupdf.Rect(50, 50, 545, 792)  # Standard Letter/A4 text box

for i, text in enumerate(PAGES_CONTENT):
    page = doc.new_page(width=595, height=842)  # A4 size in points
    # Insert title and body text
    page.insert_textbox(rect, text.strip(), fontsize=10.5, fontname="helv", lineheight=1.35)

doc.save(OUTPUT_PATH)
doc.close()

print(f"Successfully created {OUTPUT_PATH} with {len(PAGES_CONTENT)} pages!")
