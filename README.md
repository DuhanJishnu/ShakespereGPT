# Nano-Shakespeare-GPT 🎭

### Implementing Transformers from Scratch in PyTorch

Nano-Shakespeare-GPT is a simplified, character-level Generative Pre-trained Transformer (GPT) implemented entirely from scratch in PyTorch. Trained on the complete works of William Shakespeare, the model learns to mimic the style, rhythm, and vocabulary of his writing.

This project serves as a hands-on exploration of the Transformer architecture, as introduced in *“Attention Is All You Need.”* By stripping away unnecessary abstractions, it provides a transparent and educational implementation that highlights the inner workings of self-attention and deep learning models.

---


  - [Overview](https://www.google.com/search?q=%23-overview)
  - [How It Works](https://www.google.com/search?q=%23-how-it-works)
  - [Key Features](https://www.google.com/search?q=%23-key-features)
  - [Dataset](https://www.google.com/search?q=%23-dataset)
  - [Requirements](https://www.google.com/search?q=%23-requirements)
  - [Usage](https://www.google.com/search?q=%23-usage)
  - [Model Output Showcase](https://www.google.com/search?q=%23-model-output-showcase)
  - [Core Concepts & Further Reading](https://www.google.com/search?q=%23-core-concepts--further-reading)

-----

## 📖 Overview

This repository contains the code for a decoder-only Transformer model. Instead of using pre-built library functions like `nn.Transformer`, each component—from the self-attention heads to the feed-forward blocks—is implemented from the ground up. The model learns the statistical patterns of Shakespearean English at the character level and can generate new text one character at a time.

-----

## ⚙️ How It Works

The model architecture follows the standard GPT design:

1.  **Token & Positional Embeddings**: Input text is converted into a sequence of integer tokens. Each token and its position in the sequence are mapped to dense vector embeddings. These are summed to provide the model with both character identity and sequence order.
2.  **Transformer Blocks**: The core of the model consists of multiple stacked Transformer blocks (`n_layer`). Each block performs two main operations:
      * **Multi-Head Self-Attention**: This "communication" phase allows tokens to look at each other and exchange information. Each token calculates which other tokens in the context are most relevant to it. Multiple heads run this process in parallel, focusing on different aspects of the relationships.
      * **Feed-Forward Network**: This "computation" phase processes the information gathered by the attention mechanism for each token independently. It's a simple multi-layer perceptron that adds representational power.
3.  **Residual Connections & Layer Normalization**: Each sub-layer (attention, feed-forward) is wrapped with a residual connection (`x + sublayer(x)`) and followed by layer normalization. This is crucial for training deep networks by preventing vanishing/exploding gradients.
4.  **Final Output Layer**: After passing through all Transformer blocks, a final linear layer maps the processed token embeddings back to the vocabulary size, producing logits (raw scores) for the next character prediction.

-----

## ✨ Key Features

  - **Scaled Dot-Product Self-Attention**: The core mechanism enabling the model to weigh the importance of different characters in the input context.
  - **Multi-Head Attention**: Improves the model's ability to focus on different positions and representation subspaces.
  - **Positional Embeddings**: Allows the model to understand the order of characters, which is crucial for language.
  - **Character-level Tokenization**: A simple yet effective tokenizer that breaks the text down to its fundamental components.
  - **Residual Connections & Dropout**: Modern deep learning techniques for stable training and regularization.

-----

## 📚 Dataset

The model is trained on `input.txt`, which should contain the complete works of William Shakespeare. You can obtain this dataset from various sources, such as [Project Gutenberg](https://www.gutenberg.org/ebooks/100).

-----

## 💻 Requirements

The only major dependency is PyTorch.

```bash
pip install torch
```

-----

## 🚀 Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Prepare the dataset:**
    Download the Shakespeare dataset and save it as `input.txt` in the root directory of the project.

3.  **Run the training script:**

    ```bash
    python transformer.py
    ```

    The script will automatically use a CUDA-enabled GPU if `torch.cuda.is_available()` returns true; otherwise, it will fall back to the CPU. The training progress and validation loss will be printed periodically. After training is complete, the script will generate a sample of text from the trained model.

-----

Here’s the cleaned-up version without Markdown syntax, just properly formatted text:

---

## 📜 Model Output Showcase

### 🔹 Training Logs

```
step 0:    train loss 4.4753, val loss 4.4709 
step 500:  train loss 2.0840, val loss 2.1499 
step 1000: train loss 1.6609, val loss 1.8234
step 1500: train loss 1.4904, val loss 1.6784
step 2000: train loss 1.3861, val loss 1.6083
step 2500: train loss 1.3159, val loss 1.5581
step 3000: train loss 1.2624, val loss 1.5263
step 3500: train loss 1.2167, val loss 1.5004
step 4000: train loss 1.1778, val loss 1.4897
step 4500: train loss 1.1401, val loss 1.4802
```

### 🔹 Generated Sample

```
“Of the heart of himself, more!
Come, and rouse forth his spirit’s accord.
To be up-graved, those would from him—
To exert the person, that such ripe do life
In pleasing of thee. Our uncle Gloucester’s eye.”
```

### 🔹 Meaning (Approximate)

| **Original Line**                            | **Polished Line**                                | **Meaning / Interpretation**                                                              |
| -------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Of the heart of himself more!                | Of the heart of himself, more!                   | A man must draw deeper strength from within.                                              |
| Come, and no rouse forth his spit accords,   | Come, and rouse forth his spirit’s accord.       | Let him awaken his spirit in harmony.                                                     |
| To be upeth-grave Those wo'd his from him,   | To be up-graved, those would from him—           | Even those who oppose him would witness his rise (revival/persistence).                   |
| To exter person, that such ripe do life      | To exert the person, that such ripe do life      | To live fully, one must act with their true self when the moment is ripe.                 |
| In please of thee. Our uncle Gloucester's ye | In pleasing of thee. Our uncle Gloucester’s eye. | Living in a way that pleases destiny (or a beloved), under the watch of Uncle Gloucester. |


---


## 🧠 Core Concepts & Further Reading

This model is built upon several foundational concepts in deep learning. The original papers provide deep insights and are highly recommended reading.

### 1\. The Transformer & Self-Attention

The architecture is based on the decoder part of the Transformer model, which revolutionized NLP by relying solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

  - **Paper**: Vaswani, A., et al. (2017). **"Attention Is All You Need."** [**[Link to Paper]**](https://arxiv.org/abs/1706.03762)

### 2\. Residual Connections (ResNets)

Introduced to solve the degradation problem in very deep networks, residual connections allow the model to learn an identity function, ensuring that deeper layers are at least as good as shallower ones. This is the `x = x + ...` part of the `Block`'s forward pass.

  - **Paper**: He, K., et al. (2015). **"Deep Residual Learning for Image Recognition."** [**[Link to Paper]**](https://arxiv.org/abs/1512.03385)

### 3\. Dropout

A simple but powerful regularization technique. During training, it randomly sets a fraction of neuron activations to zero at each update step, preventing co-adaptation of neurons and forcing the network to learn more robust features.

  - **Paper**: Srivastava, N., et al. (2014). **"Dropout: A Simple Way to Prevent Neural Networks from Overfitting."** [**[Link to Paper]**](http://jmlr.org/papers/v15/srivastava14a.html)

### 4\. Multi-Layer Perceptron (MLP)

The "Feed-Forward Network" (`FeedForward` class) in each Transformer block is an MLP. It consists of two linear layers with a non-linear activation function (like ReLU) in between. This component processes each token's representation independently to add expressive power to the model.