# Local Document RAG Pipeline: PDF to LLM

This repository contains a complete, low-memory usage, locally hosted Retrieval-Augmented Generation (RAG) pipeline. It extracts text from PDF documents using an advanced Vision-Language Model (VLM) for OCR, embeds the extracted text into a local vector database (QDrant, in this case), and uses a quantized 3B parameter Large Language Model (LLM) to answer questions based *strictly* on your personal documents. 

The pipeline is split into two distinct stages to optimize GPU memory usage:
1. **Document Processing & OCR:** Converts PDFs to images and extracts markdown text.
2. **Embedding & RAG Inference:** Chunks text, stores it in Qdrant, and performs conversational retrieval.

# Architecture 
1. **OCR Engine ("personal_data.py")**: Uses `pdf2image` to convert PDFs into smaller JPG sections and `zai-org/GLM-OCR` to perform SOTA text recognizition, saving the output in Markdown format. 
2. **Vector Database ("rag.py")**: Locally initialize a qdrant python client to store document chunks. Allow for similarity search by using COSINE SIMILARITY and HNSW search technique. 
3. **Embedding Model ("rag.py")**: Uses `google/embeddinggemma-300m` to map text chunks into a 768-dimensional vector. Saves this embeddings to the vector database while maintaing the original content as metadata. Used to perform similarity check by COSINE SIMILARITY between user query and document information inside vector database. The model is cleared before calling the LLM engine to free up space in VRAM. 
4. **LLM Engine ("rag.py")**: Uses `Nanbeige/Nanbeige4.1-3B` quantized in 4-bit by using `bitsandbytes` to make it smaller, while maintaing efficiency. Used as the main engine to give accurate responses based on the context retrieved by the Vector Database. The model had a size of 3GB, built in a 8-bit format. To make sure it fitted with all the other processes as PyTorch overhead, CUDA context, Tokenizer, it was quantized to 4-bit size and prompt-engineered to avoid thinking process that would make the KV Cache insustainable, effectively crashing the execution of the program. 

# Prerequisites & Dependencies 

### Hardware Requirements 
* Since this project was built in a weak AI-built computer, it was necessary to quantize and effectively tune the structure to be able to fit in a casual (AI-like) setup of a 16GB RAM and 2060RTX, built in Windows 10. 

* Therefore, the requirement is basically a CUDA-compatible NVIDIA GPU with a minimum 6GB VRAM due to 4-bit quantization.

### System Dependencies
* **Poppler**: Required by `pdf2image` to read PDFs.
  * *Windows*: Download Poppler, extract it, and note the `bin` folder path (currently hardcoded to `C:\poppler\Library\bin` in the script).
  * *Linux/Mac*: Install via your package manager (e.g., `sudo apt-get install poppler-utils` or `brew install poppler`).

### Python Libraries 
* This project was built specifically for my machine and my dependencies. If you wish to replicate my setup, you will need the following Python packages installed:
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    pip install transformers accelerate bitsandbytes sentence-transformers
    pip install qdrant-client markdownify pdf2image opencv-python numpy

* Or: 
    ```bash 
    pip install requirements.txt

* Else if you want to make it to your own setup make sure you install the correct torch version based on your CUDA toolkit version. If you want to change models for stronger ones or use flash-attention, remember to install the wheels for windows, which you can find here: https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main



    
