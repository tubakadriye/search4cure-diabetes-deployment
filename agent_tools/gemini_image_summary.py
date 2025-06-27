import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO

model = genai.GenerativeModel("gemini-2.0-flash") #gemini-pro-vision

def extract_info_from_page_image(image: Image.Image, pdf_title: str, page_number: int) -> str:
    """
    Use Gemini to extract citations, figures/tables, and a summary from a page image.
    """
    #response = requests.get(gcs_url)
    #image = Image.open(BytesIO(response.content))

    prompt = f"""
    You are analyzing a scientific paper page image from the PDF titled: "{pdf_title}", page {page_number}.
    Please extract the following:
    1. All citations on the page (e.g., [1], Smith et al. 2020).
    2. Any tables or figures, including titles or captions.
    3. A brief summary of the page content.
    
    Return output in markdown with the following format:
    **Citations:** ...
    **Figures/Tables:** ...
    **Summary:** ...
    """

    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Gemini processing error: {e}"
