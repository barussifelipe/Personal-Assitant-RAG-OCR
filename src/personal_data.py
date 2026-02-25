import os
import tempfile
import numpy as np
import pdf2image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch 
import cv2 as opencv




#We need to import all the data I want to have the personal tracking it and embed it into QDrant. 

#We will use EmbeddingGemma and DeepSeekR1-QWEN8B model for the RAG. 

#Since EmbeddingGemma is text-only, we will need to convert any non-text data into text format before embedding.


output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")

def pdf_to_images(pdf_path, poppler_path):
    try:
        images = pdf2image.convert_from_path(pdf_path, size= 1024, poppler_path=poppler_path, fmt='jpeg')
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return [] 


def load_ocr_model():
    """Load the GLM-OCR model and processor once."""
    model_name = 'zai-org/GLM-OCR'
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, processor


def ocr_on_image(model, processor, image_path, page_output_path):
    """Run OCR on a single image file using GLM-OCR.
    
    Args:
        model: The loaded GLM-OCR model.
        processor: The loaded GLM-OCR processor.
        image_path: Path (string) to the image file on disk.
        page_output_path: Directory where this page's result.mmd will be saved.
    
    Returns:
        The extracted markdown text, or None on failure.
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )

        # Save result
        os.makedirs(page_output_path, exist_ok=True)
        mmd_path = os.path.join(page_output_path, "result.mmd")
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(text)

        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None


if __name__ == "__main__":
    poppler_path = r"C:\poppler\Library\bin"
     # Load model ONCE before the loop
    print("Loading OCR model...")
    model, processor = load_ocr_model()
    for file in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Input")):
        if file.lower().endswith(".pdf"):
            source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Input", file)

            images = pdf_to_images(source_path, poppler_path)
            print(f"Converted PDF to {len(images)} images.")

        
            # Create a temp directory for page images
            temp_dir = tempfile.mkdtemp(prefix="ocr_")

            file_path = os.path.join(output_path, file)
            os.makedirs(file_path, exist_ok=True)

            for i, img in enumerate(images):
                print(f"Performing OCR on image {i+1}/{len(images)}...")
                try:
                    opencv_image = opencv.cvtColor(np.array(img), opencv.COLOR_RGB2BGR)

                    # Save image to a temp file
                    temp_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
                    opencv.imwrite(temp_path, opencv_image)

                    # Each page gets its own output dir
                    page_output = os.path.join(file_path, f"page_{i+1}")

                    text = ocr_on_image(model, processor, temp_path, page_output)
                    if text is not None:
                        print(f"Page {i+1} saved to {page_output}/result.mmd")
                except Exception as e:
                    print(f"Error processing image {i+1}: {e}")
                    continue
        print(f"OCR completed. Processed {len(images)}/{len(images)} pages successfully.")
  
            

   
