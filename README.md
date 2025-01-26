# RAG: llama-rag-vector-store

This is a simple, proof-of-concept RAG app that scrapes 3 websites and returns the answer to a simple prompt based on the content of those websites. 

The prompt is hardcoded.

This code is based on a tutorial found [here](https://www.datacamp.com/tutorial/llama-3-1-rag). 

The original tutorial was written by Ryan Ong and you can find him on x [here](https://x.com/Ryan_Ong10). 

His substack is [here](https://ryanocm.substack.com/).

This RAG uses Langchain, Ollama & Llama 3.1 8B.

The websites used in this app come from [Lilian Weng's](https://lilianweng.github.io/) website. 

You can find her on x [here](https://x.com/lilianweng/).

Her github account is [here](https://github.com/lilianweng).

Differences between this app and the tutorial:
- This app uses [SentenceTransformers](https://sbert.net/) to create the embeddings instead of the OpenAI api. SentenceTransformers is free.

Python version: 3.9

Package manager: conda

## To Run
- Install Ollama
- Run the Llama model: ```ollama run llama3.1``` 
- Create a USER_AGENT env variable in .env or in bash (see the env.example file).
- Set up the app:
```
conda env create -f environment.yml
conda activate llama_rag_env_vector_store
python llama_rag.py

```

The output in your terminal will look something like this:
```
python llama_rag.py
USER_AGENT environment variable not set, consider setting it to identify your requests.
Batches: 100%|███████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 10.00it/s]
Question: What is prompt engineering?
Answer: Prompt engineering refers to methods for communicating with Large Language Models (LLMs) to steer their behavior towards desired outcomes without updating the model weights. It's an empirical science that requires experimentation and heuristics, aiming for alignment and model steerability. The goal is to optimize believability in a given context.
```
