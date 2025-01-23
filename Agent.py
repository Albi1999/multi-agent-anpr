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
import re 
from sklearn.cluster import DBSCAN

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

        # First, we need to understand which point is which
        # For this, let us first order the points by their x coordinate 
        pts_sorted_x = pts[pts[:,0].argsort()]
        # We know that points 0 and 1 are the left points, and 2 and 3 are the right points
        pair_left = pts_sorted_x[:2]
        pair_right = pts_sorted_x[2:]
        # Now we order these pairs by their y coordinate
        # The top left point will have the highest y coordinate of pair_left
        # The bottom left point will have the lowest y coordinate of pair_left
        # same idea for right pair
        pair_left_sorted_y = pair_left[pair_left[:,1].argsort()]
        pair_right_sorted_y = pair_right[pair_right[:,1].argsort()]
        rect[1] = pair_left_sorted_y[1]  # top-left
        rect[3] = pair_right_sorted_y[1] # top-right
        rect[0] = pair_left_sorted_y[0] # bottom-left
        rect[2] = pair_right_sorted_y[0] # bottom-right
    


        return rect
    
    @staticmethod
    def scale_and_resize(img, target_height=110):
        # GERMAN LICENSE PLATES : 520mmx110mm 
        """Scale and resize the license plate to the given dimensions.
        Trial for German License Plates : If cropped correctly (so just the license plate),
        I did a trial with target_height = 200, then segmenting characters and they all had
        heights around roughly 140. This means the characters take 140/200 = 0.7 of the actual height
        So if we want to do it like MNIST, they will have images of 28x28 where the actual number/letter
        is around 20x20 and the rest is padding. So my idea is to keep the image kind of big here and in
        the right aspect format (e.g. rescaling with target_height = 200, maybe change that) so that 
        we have a reference point of license plates that are in the same size, and then when we extract the
        characters, we can rescale them to 28x28 with 20x20 character and 4x4 padding."""

        # Standard height for license plates while maintaining aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0]
        target_width = int(target_height * aspect_ratio)
        
        # Resize while maintaining aspect ratio
        resized_plate = cv2.resize(img, (target_width, target_height))
        return resized_plate



    def select_corners(self,image_path):
        
        img = self.scale_and_resize(cv2.imread(image_path))
        src = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(src) < 4:
                src.append([x, y])
                # Draw circle at clicked point
                cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow('Select Corners', img_display)

        img = cv2.imread(image_path)
        img_display = img.copy()
        
        cv2.namedWindow('Select Corners')
        cv2.setMouseCallback('Select Corners', mouse_callback)
        print("Click 4 corner points. Press 'r' to reset, 'Enter' when done")

        while True:
            cv2.imshow('Select Corners', img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                src.clear()
                img_display = img.copy()
                cv2.imshow('Select Corners', img_display)
            
            elif key == ord('f') and len(src) == 4:  # f key and 4 points selected
                cv2.destroyAllWindows()
                return np.float32(src)
            
            elif key == 27:  # ESC to cancel
                cv2.destroyAllWindows()
                break


    def perspective_correction(self, image_path):


        image_path = image_path or self.IMAGE_PATH
        og_img = cv2.imread(image_path)
        src = self.select_corners(image_path)


        # Get the source points (i.e. the 4 corner points)
       # src = np.squeeze(license_cont).astype(np.float32)

        height = og_img.shape[0] 
        width = og_img.shape[1]
        # Destination points (for flat parallel)
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        license_cont = self.order_points(src)
     #   dst = self.order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(license_cont, dst)

        # Warp the image
        img_shape = (width, height)
        enhanced_img = cv2.warpPerspective(og_img, M, img_shape, flags=cv2.INTER_LINEAR)

        if not self.IS_IMAGE_PROCESSED:
            self.IMAGE_PATH = image_path.replace(".png", "_processed.png")
        cv2.imwrite(self.IMAGE_PATH, enhanced_img)
        self.IS_IMAGE_PROCESSED = True
        return self.IMAGE_PATH
    

    # TODO : understand, completely generated for now 
    def adaptive_thresholding(self, image_path=None, img=None, block_size=25, constant=1, mode='original'):
        current_thresh = None  # Store current threshold image
        
        def on_change(_):
            nonlocal current_thresh
            block = cv2.getTrackbarPos('Block Size', 'Adjust Parameters') 
            const = cv2.getTrackbarPos('Constant', 'Adjust Parameters')
            block = block * 2 + 1
            
            if image_path is not None:
                img_show = cv2.imread(image_path)
            else:
                img_show = img.copy()
                
            lab = cv2.cvtColor(img_show, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            thresh = cv2.adaptiveThreshold(
                l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block, const
            )
            current_thresh = thresh  # Update current threshold
            cv2.imshow('Output', thresh)

        cv2.namedWindow('Adjust Parameters')
        cv2.namedWindow('Output')
        print("Press 'f' when finished or 'ESC' to cancel")
        
        cv2.createTrackbar('Block Size', 'Adjust Parameters', (block_size-1)//2, 100, on_change)
        cv2.createTrackbar('Constant', 'Adjust Parameters', constant, 50, on_change)

        on_change(0)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('f'):
                cv2.destroyAllWindows()
                return current_thresh
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                break


    def character_segmentation_DBSCAN(self, image_path):
        """ Use DBSCAN to segment the characters of the license plate (after adaptive_thresholding)."""
        image_path = image_path or self.IMAGE_PATH
        # Regex to find the number in the image_path 
        img_nmb = re.search(r'\d+', image_path).group()
        img = self.scale_and_resize(cv2.imread(image_path))
        img = self.adaptive_thresholding(img=img)
        cv2.imshow('Adaptive Thresholding', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply DBSCAN






    def character_segmentation(self, image_path, kernel_size = (1,1)):
        """ Use contours on images to segment the characters of the license plate.
            Furthermore resizes the characters to 28x28 images, where the actual character is 20x20 and the rest is padding (like MNIST)."""

        image_path = image_path or self.IMAGE_PATH
        # Regex to find the number in the image_path 
        img_nmb = re.search(r'\d+', image_path).group() # since we only have one number, we can use group() to get the only match
        img = cv2.imread(image_path)
        og_img = img.copy()

        # Convert to grayscale
        gray_image = self.grayscale(img)

        # TODO : kernel potentially needs to be adjusted ; it might make characters not recognizable anymore if too big
        # also, sometimes kernel will prob. not be needed and then erosion is not necessary, since it makes the characters a bit less thick
        # so we have to figure out a way to make this more robust
        kernel = np.ones(kernel_size, np.uint8) 
        # Erosion can help separate connected components (needed sometimes for the contours, because else it draws contours over multiple characters)
        # basically makes  the characters a bit less thick (TODO : check if true)
        eroded = cv2.erode(gray_image, kernel, iterations=1)
        
     #   cv2.imshow('Eroded', eroded)
     #   cv2.waitKey(0)
     #   cv2.destroyAllWindows()

        # Gaussian Blur to prepare for Canny Detection (i.e. more robust edge detection)
        blurred = cv2.GaussianBlur(eroded, (5, 5), 0)
        # Canny Edge Detection
        canny = cv2.Canny(blurred, 100, 200)
 
        # use RETR_EXTERNAL, because with canny we else get 2 contours for each character
        contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255,0,0), 3)

        counter = 0
        for contour in sorted(contours, key=lambda x: cv2.boundingRect(x)[0]):
            x, y, w, h = cv2.boundingRect(contour)
            # width to height ratio : the images of the characters will nearly always be highr than they are wide
            # w > 80 & h > 80 : we don't want to save t–he small contours, since they are probably noise
            # TODO : hardcoded, BUT I think when the crops are more or less the same (so just the license plate) and I do the rescaling step
            # in adaptive thresholding, then it should be fine

            #The characters on GERMAN licence plates, as well as the narrow rim framing it, are black on a white background.
            #  In standard size they are 75 mm (3 in) high, and 47.5 mm (1+7⁄8 in) wide for letters or 44.5 mm (1+3⁄4 in) wide for digits
            # 47.5/75 = 0.6333, 44.5/75 = 0.5933
            if  0.57 < w/h < 0.65 and w > 50 and h > 50: 
                reg_of_interest = gray_image[y:y+h, x:x+w] # region of interest : the rectangle area that we found ; also take it from the original image!


                
                # Calculate scaling factor to fit in 20x20 box while maintaining aspect ratio
                scale = min(20.0/w, 20.0/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize character to fit in 20x20 box
                char_resized = cv2.resize(reg_of_interest, (new_w, new_h))
                
                # Create 28x28 blank (black) image
                mnist_size = np.zeros((28, 28), dtype=np.uint8)
                
                # Calculate position to center character
                x_offset = (28 - new_w) // 2
                y_offset = (28 - new_h) // 2
                
                # Place character in center of 28x28 image
                mnist_size[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_resized

                cv2.imwrite(f"VCS_Project/results/license_plates/segmented_plate.{img_nmb}/character_{counter}.png", mnist_size)
                counter += 1



         
 
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
   # agent.perspective_correction(IMAGE_PATH)
    agent.character_segmentation_DBSCAN(IMAGE_PATH)
    #agent.adaptive_thresholding(IMAGE_PATH, block_size=111, constant=2, mode='processing')
    #agent.character_segmentation(f'VCS_Project/results/license_plates/license_plate.{SAMPLE_ID}_adaptive_thresholding.png')
  #  agent.perspective_correction('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/results/license_plates/license_plate.2.png')
    


   
