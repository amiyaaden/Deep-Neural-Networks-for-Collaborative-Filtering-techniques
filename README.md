# Deep-Neural-Networks-for-Collaborative-Filtering-techniques
Collaborative Filtering Recommendation Systems - Pure NumPy Implementation
Project Description
This repository contains pure NumPy implementations of two deep learning approaches for collaborative filtering recommendation systems:

Neural Collaborative Filtering (NCF)

Graph Neural Network (GNN) for Collaborative Filtering

Both models are implemented from scratch using only NumPy, without any deep learning frameworks like PyTorch or TensorFlow. The implementations demonstrate the fundamental concepts of these approaches while being lightweight and easy to understand.

Key Features
🚀 Framework-free implementation - Only requires NumPy and basic Python

📊 Two model architectures - NCF and GNN approaches for comparison

🧠 From-scratch neural networks - Manual implementation of forward/backward passes

🌐 Graph operations without libraries - Custom sparse matrix implementation for GNN

📈 Evaluation metrics - MSE and MAE for performance comparison

Models Implemented
1. Neural Collaborative Filtering (NCF)
Combines matrix factorization with neural networks

Uses embedding layers for users and items

Processes concatenated embeddings through MLP layers

Implements manual backpropagation for training

2. Graph Neural Network (GNN)
Models user-item interactions as a bipartite graph

Implements message passing between connected nodes

Custom sparse matrix operations for efficient graph convolution

Aggregates information from multiple neighborhood depths

Dataset
The models are evaluated on the MovieLens-1M dataset, containing:

1,000,209 ratings

6,040 users

3,706 movies

Ratings on a 5-star scale
