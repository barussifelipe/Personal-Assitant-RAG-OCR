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
        images = pdf2image.convert_from_path(pdf_path, size= 1024, poppler_path=poppler_path, fmt='jpeg')
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return [] 

def side_padding(image, desired_width, fill_color=(0, 0, 0)):
    width = image.shape[1]
    if width >= desired_width:
        return image
    total_padding = desired_width - width
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    padded_image = opencv.copyMakeBorder(image, 0, 0, left_padding, right_padding, opencv.BORDER_CONSTANT, value=fill_color)
    return padded_image


def ocr_on_image(pdf_path, output_path):
    model_name = 'deepseek-ai/DeepSeek-OCR'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    
    try:
        print("Converting PDF to images...")
        images = pdf_to_images(pdf_path)
        
        for i, img in enumerate(images):
            print(f"Performing OCR on image {i+1}/{len(images)}...")
            try:
                print("Loading image...")
                with Image.open(img) as image: 
                    image.load() 
            except Exception as e:
                print(f"Error loading image: {e}")
                continue
    except Exception as e:
        print(f"Error processing images: {e}")
        return ""
        




if __name__ == "__main__":
    poppler_path = r"C:\poppler\Library\bin"
    pdf_path = r"C:\Users\benga\Documents\Documentos\Programming\PYTHON\Personal_Assistant\src\Input\relatorio-felipe-barussi.pdf"

    # images = pdf_to_images(pdf_path, poppler_path)
    # print("Converting PDF to images...")


    # for i, img in enumerate(images):
    #     print(f"Performing OCR on image {i+1}/{len(images)}...")
    #     try:
    #         print("Loading image...")
    #         opencv_image = opencv.cvtColor(np.array(img), opencv.COLOR_RGB2BGR)
    #         model_image = side_padding(opencv_image, desired_width=1024, fill_color=(0, 0, 0))
    #         print(f"Displaying image with shape:", {np.array(img).shape})
    #         window_name = f"Image {i+1}"
    #         opencv.imshow(window_name, opencv_image)
    #         opencv.waitKey(0)
    #         opencv.destroyAllWindows()
    #     except Exception as e:
    #         print(f"Error loading image: {e}")
    #         continue
    print(torch.cuda.is_available())
            

   
