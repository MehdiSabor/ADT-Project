"""
Main entry point for the K-Means clustering analysis project.

This module loads three datasets (Iris, AI Index, Earthquakes), preprocesses them
with standardization, runs the alternate K-Means algorithm for demonstration,
and generates comparison plots (cluster visualizations, convergence curves,
reassignment counts, and runtime vs. k).

Usage:
    Run from project root: python main.py
    Requires: data/iris.csv, data/ai_index.csv, data/earthquakes.csv
    Output: plots saved to the plots/ directory.
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from kmeans_alternate import KMeansAlternate
from plots import (
    plot_clusters,
    plot_convergence,
    plot_reassigned,
    plot_runtime,
)


def load_datasets():
    """
    Load the three project datasets from CSV files.

    Returns:
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame): Raw DataFrames
            for Iris, AI Index, and Earthquakes in that order.
    """
    iris = pd.read_csv("data/iris.csv")
    ai = pd.read_csv("data/ai_index.csv")
    eq = pd.read_csv("data/earthquakes.csv")
    return iris, ai, eq


def preprocess_iris(df):
    """
    Preprocess the Iris dataset for clustering.

    Drops the species column (label), removes missing values, and standardizes
    features to zero mean and unit variance.

    Args:
        df: Raw Iris DataFrame (must include 'species' column).

    Returns:
        np.ndarray: Scaled numeric array of shape (n_samples, n_features).
    """
    # Remove label column so we cluster on features only
    df = df.drop(columns=["species"])
    df = df.dropna()
    scaler = StandardScaler()
    return scaler.fit_transform(df)


def preprocess_ai(df):
    """
    Preprocess the AI Index dataset for clustering.

    Keeps only numeric columns, drops rows with missing values, and
    standardizes features.

    Args:
        df: Raw AI Index DataFrame.

    Returns:
        np.ndarray: Scaled numeric array of shape (n_samples, n_features).
    """
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    scaler = StandardScaler()
    return scaler.fit_transform(df)


def preprocess_eq(df):
    """
    Preprocess the Earthquakes dataset for clustering.

    Keeps only magnitude, depth, latitude, and longitude; drops missing
    values and standardizes.

    Args:
        df: Raw Earthquakes DataFrame.

    Returns:
        np.ndarray: Scaled numeric array of shape (n_samples, 4).
    """
    df = df[["magnitude", "depth", "latitude", "longitude"]]
    df = df.dropna()
    scaler = StandardScaler()
    return scaler.fit_transform(df)


if __name__ == "__main__":
    # Load and preprocess all datasets
    iris, ai, eq = load_datasets()
    iris_data = preprocess_iris(iris)
    ai_data = preprocess_ai(ai)
    eq_data = preprocess_eq(eq)

    print("Iris shape:      ", iris_data.shape)
    print("AI Index shape:  ", ai_data.shape)
    print("Earthquake shape:", eq_data.shape)

    # Demo: fit alternate K-Means on Iris with k=2
    model = KMeansAlternate(k=2)
    model.fit(iris_data)
    print("Iterations:", model.n_iterations)
    print("Final SSE: ", model.sse_history[-1])
    print("Labels:    ", model.labels)

    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    datasets = {
        "Iris": iris_data,
        "AI_Index": ai_data,
        "Earthquakes": eq_data,
    }

    for name, data in datasets.items():
        print(f"\nGenerating plots for {name}...")
        plot_clusters(data, name)
        plot_convergence(data, name)
        plot_reassigned(data, name)
        plot_runtime(data, name)

    print("\nAll plots saved in /plots folder!")
