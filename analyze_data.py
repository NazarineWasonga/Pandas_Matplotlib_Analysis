"""
Analyzing Data with Pandas and Visualizing Results with Matplotlib
Assignment: Data Analysis and Visualization
Dataset: Iris (from sklearn.datasets)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


def main():
    # ---------------------------------------------------------------
    # Task 1: Load and Explore the Dataset
    # ---------------------------------------------------------------
    try:
        # Load iris dataset
        iris = load_iris(as_frame=True)
        df = iris.frame  # Convert to pandas DataFrame
        df["species"] = iris.target_names[iris.target]

        print("✅ Dataset loaded successfully.\n")

        # Display first few rows
        print("First 5 rows of dataset:")
        print(df.head(), "\n")

        # Dataset info
        print("Dataset Info:")
        print(df.info(), "\n")

        # Check for missing values
        print("Missing values per column:")
        print(df.isnull().sum(), "\n")

        # Clean dataset (drop missing values if any)
        df = df.dropna()

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # ---------------------------------------------------------------
    # Task 2: Basic Data Analysis
    # ---------------------------------------------------------------
    print("Basic Statistics of Dataset:\n")
    print(df.describe(), "\n")

    # Grouping: average petal length per species
    grouped = df.groupby("species")["petal length (cm)"].mean()
    print("Average petal length per species:")
    print(grouped, "\n")

    # Observation
    print("Observation: Iris-virginica tends to have the largest petal length.\n")

    # ---------------------------------------------------------------
    # Task 3: Data Visualization
    # ---------------------------------------------------------------
    sns.set(style="whitegrid")  # Better plot style

    # 1. Line Chart: Sepal length trend across dataset index
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color="blue")
    plt.title("Line Chart of Sepal Lengths")
    plt.xlabel("Index")
    plt.ylabel("Sepal Length (cm)")
    plt.legend()
    plt.show()

    # 2. Bar Chart: Average petal length per species
    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar", color=["skyblue", "orange", "green"])
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.show()

    # 3. Histogram: Distribution of sepal width
    plt.figure(figsize=(8, 5))
    plt.hist(df["sepal width (cm)"], bins=15, color="purple", edgecolor="black")
    plt.title("Histogram of Sepal Width")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Frequency")
    plt.show()

    # 4. Scatter Plot: Sepal length vs Petal length
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="sepal length (cm)",
        y="petal length (cm)",
        hue="species",
        data=df,
        palette="deep"
    )
    plt.title("Scatter Plot: Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.show()


if __name__ == "__main__":
    main()
