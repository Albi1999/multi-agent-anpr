from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from PIL import Image, ImageEnhance
import ollama
import base64
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class LicensePlateAgent:
    def __init__(self, llm_name="llama3.2-vision", temperature=0):
        self.IMAGE_PATH = ""
        self.IS_IMAGE_PROCESSED = False
        self.VISION_MODEL = llm_name
        self.CONTEXT_PROMPT = '''
            The image is a car plate photo.
            The output should be the car plate.
            Output should be in this format - <Number of Car Plate> - Do not output anything else.
            '''

        # Initialize the LLM
        self.llm = ChatOllama(model=llm_name, temperature=temperature)

        # Define tool functions
        self.tools = {
            "grayscale": self.grayscale_tool,
            "adjust_exposure": self.adjust_exposure_tool,
            "adjust_brilliance": self.adjust_brilliance_tool,
            "adjust_contrast": self.adjust_contrast_tool,
            "adjust_sharpness": self.adjust_sharpness_tool,
            "edge_detection": self.edge_detection_tool,
        }

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_image

    def suggest_tool(self, image_path=None) -> str:
        """Suggests a tool based on the given image."""
        image_path = image_path or self.IMAGE_PATH
        encoded_image = self.encode_image(image_path)

        MESSAGES = [{
            "role": "system", 
            "content": 
                '''
                f' Tools: "grayscale", "adjust_exposure", "adjust_brilliance", "adjust_contrast", "adjust_sharpness", "edge_detection"
                '''
        },
        {
            "role": "user",
            "content": """
            Based on the image data and the context below, suggest the best tool and its input to improve the image readability.
            Context: {context}

            Respond in the format: ToolName(InputValue).
            """.format(context=self.CONTEXT_PROMPT),
            "images": [encoded_image]
        }]

        response = ollama.chat(
            model=self.VISION_MODEL,
            messages=MESSAGES
        )

        # Parse the response to extract tool name and input
        tool_name, tool_input = self.parse_tool_response(response['message']['content'])
        return print(f"Tool: {tool_name}, Input: {tool_input}")

    @staticmethod
    def parse_tool_response(response: str) -> tuple:
        """Parses the tool response to extract tool name and input."""
        # Assuming the response is structured as "ToolName(InputValue)"
        if "(" in response and ")" in response:
            tool_name = response.split("(")[0]
            tool_input = response.split("(")[1].split(")")[0]
            return tool_name, tool_input
        return response, ""

    def vision_model_tool(self, image_path=None) -> str:
        """Perform OCR on the given image using the Vision Model."""
        image_path = image_path or self.IMAGE_PATH
        encoded_image = self.encode_image(image_path)
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

    def show_image(self, image_path=None):
        """Displays the image at the given path."""
        image_path = image_path or self.IMAGE_PATH
        img = Image.open(image_path)
        img.show()

    # Image enhancement methods
    @staticmethod
    def grayscale(image):
        """Convert an image to grayscale."""
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



    @staticmethod 
    def order_points(pts):
        """Order points in clockwise order starting from top-left"""
        # Initialize ordered coordinates
        rect = np.zeros((4, 2), dtype=np.float32)
        # TODO : check if this universally works.. , I don't know 
        s = pts.sum(axis=1)
        rect[1] = pts[np.argmin(s)]  # top-right
        rect[3] = pts[np.argmax(s)]  # bottom-left

        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(diff)]  # top-left
        rect[2] = pts[np.argmax(diff)]  # bottom-right

        return rect



    def contour_detection(self, image_path):
        """ Detect contours of an image (i.e. contours of the license plate)."""
        # Inspired by :
        # https://stackoverflow.com/questions/62295185/warping-a-license-plate-image-to-be-frontal-parallel
        # in OpenCV, finding contours is like finding white objects from a black background, thereore object should be white and background black
        # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        # Therefore, first we need to convert the image to grayscale
        image_path = image_path or self.IMAGE_PATH
        img = cv2.imread(image_path)
        og_img = img.copy()
        gray_image = self.grayscale(img)
        contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Probably the correct contour (i.e. the one surrounding the whole license plate), should be the one that has the largest area
        biggest_area = 0 
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # This we use to get the 4 corner points of our main contour (i.e. the one surrounding the whole license plate)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour,0.02*peri, True)
            if area > biggest_area and len(approx) == 4:
                biggest_area = area
                license_cont  = approx # This stores the 4 corner points
                biggest_contour_idx = idx # this index we need to then draw the contour onto the image

        # Next, we want to warp that contour such that the license plat is flat and frontal parallel
        if biggest_contour_idx is not None: # Check we actually found a contour
       #     cv2.drawContours(img, contours, biggest_contour_idx, (255,0,0), 3)
       #     cv2.imshow('Contour', img)
       #     cv2.waitKey(0)
       #     cv2.destroyAllWindows()

            # Get the source points (i.e. the 4 corner points)
            src = np.squeeze(license_cont).astype(np.float32)

            height = og_img.shape[0] 
            width = og_img.shape[1]
            # Destination points (for flat parallel)
            dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

            # Order the points correctly
            license_cont = self.order_points(src)
            dst = self.order_points(dst)

            # Get the perspective transform
            M = cv2.getPerspectiveTransform(src, dst)

            # Warp the image
            img_shape = (width, height)
            enhanced_img = cv2.warpPerspective(og_img, M, img_shape, flags=cv2.INTER_LINEAR)

            if not self.IS_IMAGE_PROCESSED:
                self.IMAGE_PATH = image_path.replace(".png", "_processed.png")
            cv2.imwrite(self.IMAGE_PATH, enhanced_img)
            self.IS_IMAGE_PROCESSED = True
            return self.IMAGE_PATH
     


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

    def adjust_brilliance_tool(self, factor: float = 1.0, image_path=None) -> str:
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
    SAMPLE_ID = 2
    IMAGE_PATH = f'results/license_plates/license_plate.{SAMPLE_ID}.png'
    IMAGE_PATH = f'VCS_Project/results/license_plates/license_plate.{SAMPLE_ID}.png' # TODO : uncomment, need it bc I am working on a virtual environment and don't want do add it to git 
    agent = LicensePlateAgent()
    agent.contour_detection(IMAGE_PATH)


   
