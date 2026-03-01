import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from plots import (plot_clusters, plot_convergence, 
                   plot_reassigned, plot_runtime)

from kmeans_alternate import KMeansAlternate


def load_datasets():
    iris = pd.read_csv('data/iris.csv')
    ai   = pd.read_csv('data/ai_index.csv')
    eq   = pd.read_csv('data/earthquakes.csv')
    return iris, ai, eq

def preprocess_iris(df):
    # Drop the species text column
    df = df.drop(columns=['species'])
    df = df.dropna()
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def preprocess_ai(df):
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def preprocess_eq(df):
    # Keep only the most relevant numeric columns
    df = df[['magnitude', 'depth', 'latitude', 'longitude']]
    df = df.dropna()
    scaler = StandardScaler()
    return scaler.fit_transform(df)

if __name__ == '__main__':
    iris, ai, eq = load_datasets()

    iris_data = preprocess_iris(iris)
    ai_data   = preprocess_ai(ai)
    eq_data   = preprocess_eq(eq)

    print("Iris shape:      ", iris_data.shape)
    print("AI Index shape:  ", ai_data.shape)
    print("Earthquake shape:", eq_data.shape)


 
model = KMeansAlternate(k=2)
model.fit(iris_data)
print("Iterations:", model.n_iterations)
print("Final SSE: ", model.sse_history[-1])
print("Labels:    ", model.labels)

# Create plots folder
os.makedirs('plots', exist_ok=True)

datasets = {
    'Iris':        iris_data,
    'AI_Index':    ai_data,
    'Earthquakes': eq_data
}

for name, data in datasets.items():
    print(f"\nGenerating plots for {name}...")
    plot_clusters(data, name)
    plot_convergence(data, name)
    plot_reassigned(data, name)
    plot_runtime(data, name)

print("\nAll plots saved in /plots folder!")