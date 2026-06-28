# 🩺 Medical Chatbot using BERT

A medical chatbot built with BERT embeddings and semantic search using FAISS.

## Features

- Medical Question Answering
- BERT Sentence Embeddings
- FAISS Semantic Search
- Streamlit Interface
- HuggingFace Dataset (MedQuAD)

## Tech Stack

- Python
- PyTorch
- Transformers
- Sentence Transformers
- FAISS
- Streamlit

## Dataset

MedQuAD

## Installation

```bash
git clone https://github.com/USERNAME/medical-chatbot-bert.git

cd medical-chatbot-bert-fine-tuning

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

Generate embeddings

```bash
python src/embeddings.py
```
Run the retrievla script which will generate the FAISS index

```bash
python src/retrieval.py
```
Run chatbot

```bash
streamlit run app/app.py
```
if that doesn't work, try : 

```bash
Python -m streamlit run app.py
```
