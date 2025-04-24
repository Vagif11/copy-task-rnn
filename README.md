# copy-task-rnn
Comparative analysis of LSTM, GRU, and their multiplicative variants on sequence modeling


# 🧠 Copy Task using LSTMs and GRUs (Standard vs Multiplicative)

Hi! I'm Vagif Asadov, an international student studying Computer Science, passionate about NLP and deep learning.  
This project compares different recurrent neural architectures on a synthetic **sequence memorization task** known as the **copy task**.

---

## 💡 What’s the Copy Task?

The model receives a sequence of random tokens, followed by a **delimiter (0)**, and must **reproduce** the original sequence after the delimiter.

🧠 It tests the model’s ability to:
- Memorize sequences
- Handle long-term dependencies
- Generalize to **longer sequences** than seen during training

---

## 📊 Models Compared

| Model Type            | Description                                      |
|----------------------|--------------------------------------------------|
| `Standard LSTM`       | Vanilla LSTM with standard gates                |
| `Multiplicative LSTM` | Adds a memory mechanism to modulate inputs      |
| `Standard GRU`        | Gated Recurrent Unit – lighter than LSTM        |
| `Multiplicative GRU`  | GRU with multiplicative memory enhancement      |

All models were trained on sequences of length **T=100**, and tested on lengths up to **T=1000**.

---

## 📈 Results (Summary)

| Model               | Accuracy @ T=1000 |
|---------------------|-------------------|
| Standard LSTM       | ~55.5%            |
| Multiplicative LSTM | ~55.5%            |
| Standard GRU        | ~55.5%            |
| Multiplicative GRU  | ~55.6%            |

Multiplicative models showed slightly better generalization at longer lengths but not drastically.

---

## 🔍 Key Insights

- **GRUs converged faster** than LSTMs.
- Multiplicative memory helped **slightly on longer sequences**.
- No catastrophic forgetting—models were stable when tested on unseen sequence lengths.

---

## 📁 Project Structure

Skills Practiced
Manual LSTM/GRU implementation (no nn.LSTM)

Weight initialization (Xavier)

Sequence modeling

Training & evaluation loops in PyTorch

Comparing generalization performance across architectures

Inspired by concepts from:

MIT Deep Learning (Lex Fridman, 6.S191)

Stanford CS224n: NLP with Deep Learning

my linkedin - https://www.linkedin.com/in/asadovagif/ 
