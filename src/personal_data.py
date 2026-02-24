from qdrant_client import QdrantClient 
import ollama
import os
import tempfile
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
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")

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


def load_ocr_model():
    """Load the OCR model and tokenizer once."""
    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model = model.eval()
    return model, tokenizer


def ocr_on_image(model, tokenizer, image_path, page_output_path):
    """Run OCR on a single image file using an already-loaded model.
    
    Args:
        image_path: Path (string) to the image file on disk.
        page_output_path: Directory where this page's result.mmd will be saved.
    
    Returns:
        The extracted markdown text, or None on failure.
    """
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    try:
        model.infer(tokenizer, prompt=prompt, image_file=image_path, output_path=page_output_path, base_size=1024, image_size=640, crop_mode=True, save_results=True, test_compress=True)
        
        # infer() saves to result.mmd but returns None — read it back
        mmd_path = os.path.join(page_output_path, "result.mmd")
        if os.path.exists(mmd_path):
            with open(mmd_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None
        




if __name__ == "__main__":
    poppler_path = r"C:\poppler\Library\bin"
    pdf_path = r"C:\Users\benga\Documents\Documentos\Programming\PYTHON\Personal_Assistant\src\Input\relatorio-felipe-barussi.pdf"

    images = pdf_to_images(pdf_path, poppler_path)
    print(f"Converted PDF to {len(images)} images.")

    # Load model ONCE before the loop
    print("Loading OCR model...")
    model, tokenizer = load_ocr_model()

    # Create a temp directory for padded images
    temp_dir = tempfile.mkdtemp(prefix="ocr_")
    os.makedirs(output_path, exist_ok=True)

    results = []
    for i, img in enumerate(images):
        print(f"Performing OCR on image {i+1}/{len(images)}...")
        try:
            opencv_image = opencv.cvtColor(np.array(img), opencv.COLOR_RGB2BGR)
            model_image = side_padding(opencv_image, desired_width=1024, fill_color=(0, 0, 0))

            # Save padded image to a temp file (infer() expects a file path)
            temp_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
            opencv.imwrite(temp_path, model_image)

            # Each page gets its own output dir so result.mmd isn't overwritten
            page_output = os.path.join(output_path, f"page_{i+1}")
            os.makedirs(page_output, exist_ok=True)

            text = ocr_on_image(model, tokenizer, temp_path, page_output)
            if text is not None:
                results.append(text)
                print(f"Page {i+1} saved to {page_output}/result.mmd")
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")
            continue

    # Combine all pages into a single markdown file
    if results:
        combined_path = os.path.join(output_path, "full_document.md")
        with open(combined_path, "w", encoding="utf-8") as f:
            for i, page_text in enumerate(results):
                f.write(f"\n\n<!-- Page {i+1} -->\n\n")
                f.write(page_text)
        print(f"Full document saved to {combined_path}")

    print(f"OCR completed. Processed {len(results)}/{len(images)} pages successfully.")
  
            

   
