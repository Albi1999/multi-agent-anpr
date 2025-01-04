#from langchain.agents import initialize_agent, Tool
#from langchain.tools import Tool
#from langchain_ollama import ChatOllama
#import easyocr
#import pytesseract
from PIL import Image, ImageEnhance
import ollama
import base64
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"



class LicensePlateAgent:
    def __init__(self, llm_name="llama3.2:1b", temperature=0):
        self.IMAGE_PATH = ""
        self.IS_IMAGE_PROCESSED = False
        self.VISION_MODEL = "llama3.2-vision"
        self.CONTEXT_PROMPT = '''
            The image is a car plate photo.
            The output should be the car plate.
            Output should be in this format - <Number of Car Plate> - Do not output anything else.
            '''

        '''TODO: 
        # Initialize the LLM
        self.llm = ChatOllama(model=llm_name, temperature=temperature)
        
        # Define tools
        
        self.tools = [
            Tool(
                name="easyOCR",
                func=self.easyOCR_tool,
                description="Extracts text from an image using OCR."
            ),
            Tool(
                name="Vision Model",
                func=self.vision_model_tool,
                description="Extracts text from an image using a vision model."
            ),
            Tool(
                name="Pytesseract",
                func=self.pytesseract_tool,
                description="Extracts text from an image using pytesseract."
            ),
        ]

        # Initialize LangChain agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type="zero-shot-react-description"
        )
        '''

    def vision_model_tool(self, image_path=None) -> str:
        """Perform OCR on the given image using Llama 3.2-Vision."""
        image_path = image_path or self.IMAGE_PATH
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        MESSAGES = [{
            "role": "user",
            "content": self.CONTEXT_PROMPT,
            "images": [encoded_image]
        }]
        response = ollama.chat(
            model=self.VISION_MODEL,
            messages=MESSAGES
        )

        self.IMAGE_PATH = image_path
        return response['message']['content'].strip()

    '''TODO
    def easyOCR_tool(self, image_path=None) -> str:
        """Performs OCR on an image."""
        image_path = image_path or self.IMAGE_PATH
        if not image_path:
            raise ValueError("No image path provided.")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image from path.")


        reader = easyocr.Reader(['en'])
        extracted_text = reader.readtext(image)

        print("OCR results:", extracted_text)  
        self.IMAGE_PATH = image_path
        return extracted_text

    def pytesseract_tool(self, image_path=None) -> str:
        """Performs OCR on an image."""
        image_path = image_path or self.IMAGE_PATH
        if not image_path:
            raise ValueError("No image path provided.")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image from path.")


        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(image, config=custom_config)

        print("OCR results:", extracted_text)  # Debugging output
        self.IMAGE_PATH = image_path
        return extracted_text.strip()
    '''

    def show_image(self, image_path=None):
        """Displays the image at the given path."""
        image_path = image_path or self.IMAGE_PATH
        img = Image.open(image_path)
        img.show()

    '''
    def automatic_mode(self, image_path: str, text_prompt: str = "") -> str:
        """Runs the full pipeline: enhance image, OCR, and LLM."""
        # Use LangChain agent to run the tools and LLM
        task = f"""
        The goal is to extract the license plate number from the image at {image_path}.
        First, try to extract the license plate number without using the OCR tool.
        If you can successfully extract the license plate, stop the process and output the result.
        If you cannot extract the license plate, use the tool calling the function OCR() with input = {image_path} and return the output of the tool.
        After that, output the final result.
        """
        return self.agent.invoke(task)
    '''

    # Image enhancement methods
    @staticmethod
    def grayscale(image):
        """Colorize a grayscale image."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        return thresh_image
    
    @staticmethod
    def adjust_exposure(image, factor):
        """Adjust the exposure of an image by modifying its brightness."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_brilliance(image, factor):
        """Adjust the brilliance of an image (similar to exposure)."""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_contrast(image, factor):
        """Adjust the contrast of an image."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_sharpness(image, factor):
        """Adjust the sharpness of an image."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def edge_detection(image):
        """Detect edges in an image using the Canny edge detection algorithm."""
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edges = cv2.Canny(bfilter, 100, 200)
        
        return edges


    def adjust_exposure_tool(self, factor: float = 1.0, image_path=None) -> str:
        """Adjusts the exposure of an image and saves it."""
        image_path = image_path or self.IMAGE_PATH
        img = Image.open(image_path)
        enhanced_img = self.adjust_exposure(img, factor)
        if not self.IS_IMAGE_PROCESSED:
            self.IMAGE_PATH = image_path.replace(".png", "_processed.png")
        enhanced_img.save(self.IMAGE_PATH)
        self.IS_IMAGE_PROCESSED = True
        return self.IMAGE_PATH

    def adjust_brilliance_tool(self, factor: float = 1.0,  image_path=None,) -> str:
        """Adjusts the brilliance of an image and saves it."""
        image_path = image_path or self.IMAGE_PATH
        img = Image.open(image_path)
        enhanced_img = self.adjust_brilliance(img, factor)
        if not self.IS_IMAGE_PROCESSED:
            self.IMAGE_PATH = image_path.replace(".png", "_processed.png")
        enhanced_img.save(self.IMAGE_PATH)
        self.IS_IMAGE_PROCESSED = True
        return self.IMAGE_PATH

    def adjust_contrast_tool(self, factor: float = 1.0, image_path=None) -> str:
        """Adjusts the contrast of an image and saves it."""
        image_path = image_path or self.IMAGE_PATH
        img = Image.open(image_path)
        enhanced_img = self.adjust_contrast(img, factor)
        if not self.IS_IMAGE_PROCESSED:
            self.IMAGE_PATH = image_path.replace(".png", "_processed.png")
        enhanced_img.save(self.IMAGE_PATH)
        self.IS_IMAGE_PROCESSED = True
        return self.IMAGE_PATH

    def adjust_sharpness_tool(self, factor: float = 1.0, image_path=None) -> str:
        """Adjusts the sharpness of an image and saves it."""
        image_path = image_path or self.IMAGE_PATH
        img = Image.open(image_path)
        enhanced_img = self.adjust_sharpness(img, factor)
        if not self.IS_IMAGE_PROCESSED:
            self.IMAGE_PATH = image_path.replace(".png", "_processed.png")
        enhanced_img.save(self.IMAGE_PATH)
        self.IS_IMAGE_PROCESSED = True
        return self.IMAGE_PATH

    def edge_detection_tool(self, image_path=None) -> str:
        """Detects edges in an image and saves the result."""
        image_path = image_path or self.IMAGE_PATH
        img = Image.open(image_path)
        edges = self.edge_detection(img)
        if not self.IS_IMAGE_PROCESSED:
            self.IMAGE_PATH = image_path.replace(".png", "_processed.png")
        cv2.imwrite(self.IMAGE_PATH, edges)
        self.IS_IMAGE_PROCESSED = True
        return self.IMAGE_PATH
    
    def grayscale_tool(self, image_path=None) -> str:
        """Converts an image to grayscale and saves it."""
        image_path = image_path or self.IMAGE_PATH
        img = cv2.imread(image_path)
        processed_image = self.grayscale(img)
        if not self.IS_IMAGE_PROCESSED:
            self.IMAGE_PATH = image_path.replace(".png", "_processed.png")
        cv2.imwrite(self.IMAGE_PATH, processed_image)
        self.IS_IMAGE_PROCESSED = True
        return self.IMAGE_PATH
    
    def restart(self):
        """Restart the agent by clearing the image path."""
        self.IMAGE_PATH = self.IMAGE_PATH.replace("_processed", "")
        self.IS_IMAGE_PROCESSED = False


# Example usage
if __name__ == "__main__":
    SAMPLE_ID = 1
    IMAGE_PATH = f'results/license_plates/license_plate.{SAMPLE_ID}.png'
    agent = LicensePlateAgent()
