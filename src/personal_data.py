from qdrant_client import QdrantClient 
import ollama
import os
import numpy as np
import pdf2image
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch 
import cv2 as opencv

#We need to import all the data I want to have the personal tracking it and embed it into QDrant. 

#We will use EmbeddingGemma and DeepSeekR1-QWEN8B model for the RAG. 

#Since EmbeddingGemma is text-only, we will need to convert any non-text data into text format before embedding.

qdrant_client = QdrantClient(url="http://localhost:6333")
output_path = "../src/Output"

def pdf_to_images(pdf_path, poppler_path):
    try:
        images = pdf2image.convert_from_path(pdf_path, size= (1024, 1024), poppler_path=poppler_path)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return [] 
    
# def ocr_on_image(pdf_path, output_path):
#     model_name = 'deepseek-ai/DeepSeek-OCR'

#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
#     model = model.eval().cuda().to(torch.bfloat16)

#     prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    
    
#     try:
#         print("Converting PDF to images...")
#         images = pdf_to_images(pdf_path)
        
#         for i, img in enumerate(images):
#             print(f"Performing OCR on image {i+1}/{len(images)}...")
#             try:
#                 print("Loading image...")
#                 with Image.open(img) as image: 
#                     image.load() 
#             except Exception as e:
#                 print(f"Error loading image: {e}")
#                 continue
#     except Exception as e:
#         print(f"Error processing images: {e}")
#         return ""
        

# print("Converting PDF to images...")

poppler_path = r"C:\poppler\Library\bin"

pdf_path = r"C:\Users\benga\Documents\Documentos\Programming\PYTHON\Personal_Assistant\src\Input\relatorio-felipe-barussi.pdf"
images = pdf_to_images(pdf_path, poppler_path)

for i, img in enumerate(images):
    print(f"Performing OCR on image {i+1}/{len(images)}...")
    try:
        print("Loading image...")
        print(f"Displaying image with shape:", {np.array(img).shape})
        opencv_image = opencv.cvtColor(np.array(img), opencv.COLOR_RGB2BGR)
        window_name = f"Image {i+1}"
        opencv.imshow(window_name, opencv_image)
        opencv.waitKey(0)
        opencv.destroyAllWindows()
    except Exception as e:
        print(f"Error loading image: {e}")
        continue
    
            

   
