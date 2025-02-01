#from langchain_core.messages import HumanMessage
#from langchain_ollama import ChatOllama
from PIL import Image, ImageEnhance
from pathlib import Path 
from functools import wraps
import ollama
import base64
import cv2
import numpy as np
import warnings
import os
import re

warnings.filterwarnings("ignore")


class LicensePlateAgent:
    def __init__(self, image_path, llm_name="llama3.2-vision", verbose=True):
        print("Hi, I'm Plately!\nHow can I help you?")
        self.VISION_MODEL = llm_name
        self.STARTING_IMAGE_PATH = image_path
        self.TOOLS = {
            "Grayscale": self.grayscale_tool,
            "Adjust Exposure": self.adjust_exposure_tool,
            "Adjust Brilliance": self.adjust_brilliance_tool,
            "Adjust Contrast": self.adjust_contrast_tool,
            "Adjust Sharpness": self.adjust_sharpness_tool,
            "Edge Detection": self.edge_detection_tool,
            "Adaptive Thresholding": self.adaptive_thresholding,
            "Invert Colors": self.invert_tool,
            "Paint Image": self.paint_image
        }
        
        self.current_image_path = image_path
        self.step = 0
        self.max_steps = 0
        self.verbose = verbose

         # Initialize the LLM with the tools
        #self.llm = ChatOllama(model=llm_name, temperature=temperature)
        
        
    def process(func):
        """Decorator to update the step counter after each image processing method."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function
            result = func(*args, **kwargs)
            # Update the step counter
            args[0].update_step()
            # If the function has a verbose argument, print the image path
            if args[0].verbose:
                print(f"Current Step: {args[0].step}\nCurrent image path: {args[0].current_image_path}")

            
            return result
        
        return wrapper


    def suggest_tool(self) -> str:
        """Suggests a tool based on the given image."""
        encoded_image = self.encode_image(self.current_image_path)
        context = "The image is a car plate photo."
        MESSAGES = [{
            "role": "system", 
            "content": 
                '''
                You are an expert in image processing and have been asked to suggest a tool to improve the image readability.
                You have the following tools at your disposal:
                Tools: {tools}
                '''
        },
        {
            "role": "user",
            "content": """
            Based on the image data and the context below, suggest the best tool and its input to improve the image readability.
            Context: {context}

            Respond in the format: ToolName(InputValue).
            """.format(context=context, tools=", ".join(self.TOOLS.keys())),
            "images": [encoded_image]
        }]

        response = ollama.chat(
            model=self.VISION_MODEL,
            messages=MESSAGES
        )

        # Parse the response to extract tool name and input
        tool_name, tool_input = self.parse_tool_response(response['message']['content'])
        return print(f"Tool: {tool_name}, Input: {tool_input}")
    
    
    ''' TODO : implement this method
    def suggest_tool_llm(self, image_path=None) -> str:
        """Suggests a tool based on the given image using the LLM."""
        image_path = image_path or self.current_image_path
        encoded_image = self.encode_image(image_path)
        context = "The image is a car plate photo."
        system_prompt = 
                        You are an expert in image processing and have been asked to suggest a tool to improve the image readability.
                        You have the following tools at your disposal:
                        Tools: {tools}
                        .format(tools=", ".join(self.tools.keys()))
        user_prompt = """
                    Based on the image data and the context below, suggest the best tool and its input to improve the image readability.
                    Context: {context}
                    """.format(context=context)
                    

        MESSAGES = [
            HumanMessage(
                role="system",
                content=[
                    {"type": "text", "text": system_prompt},
                    
                ]
            ),
            HumanMessage(
                role="user",
                content=[
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "url": f'data:image/png;base64,{encoded_image}'}
                ]
            )
        ]

        response = self.llm.invoke(messages=MESSAGES)
    '''
    
    
    
    
    
    ############ Tool response parsing ############
    @staticmethod
    def parse_tool_response(response: str) -> tuple:
        """Parses the tool response to extract tool name and input."""
        # Assuming the response is structured as "ToolName(InputValue)"
        if "(" in response and ")" in response:
            tool_name = response.split("(")[0]
            tool_input = response.split("(")[1].split(")")[0]
            return tool_name, tool_input
        return response, ""
    
    ############ Image encoding  ############
    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_image

    ############ Image processing tools ############
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
        """Scale and resize the license plate to the given dimensions."""

        # Standard height for license plates while maintaining aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0]
        target_width = int(target_height * aspect_ratio)
        
        # Resize while maintaining aspect ratio
        resized_plate = cv2.resize(img, (target_width, target_height))
        return resized_plate
    


    ############ Adaptive Threshold  #############
    
    def select_corners(self,image):
        
       # img = self.scale_and_resize(cv2.imread(image))
        src = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(src) < 4:
                src.append([x, y])
                # Draw circle at clicked point
                cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow('Select Corners', img_display)

       # img = image
        img_display = image.copy()
        
        cv2.namedWindow('Select Corners')
        cv2.setMouseCallback('Select Corners', mouse_callback)
        print("Click 4 corner points. Press 'r' to reset, 'Enter' when done")

        while True:
            cv2.imshow('Select Corners', img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                src.clear()
                img_display = image.copy()
                cv2.imshow('Select Corners', img_display)
            
            elif key == ord('f') and len(src) == 4:  # f key and 4 points selected
                cv2.destroyAllWindows()
                return np.float32(src)
            
            elif key == 27:  # ESC to cancel
                cv2.destroyAllWindows()
                break
            
            
    def perspective_correction(self, img=None, image_path=None):


        og_img = img.copy()

        src = self.select_corners(og_img)


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

        return enhanced_img
    

    def adaptive_thresholding(self, image_path=None, img=None, block_size=25, constant=1, mode='processing'):
        current_thresh = None  # Store current threshold image

        if mode == 'saving':
            # Get the name of the image (for saving purposes)
            image_path_splitted = image_path.split('/')
            name = image_path_splitted[-1]
        
        def on_change(_):
            nonlocal current_thresh
            block = cv2.getTrackbarPos('Block Size', 'Adjust Parameters') 
            const = cv2.getTrackbarPos('Constant', 'Adjust Parameters')
            block = block * 2 + 1 # always keep block size an uneven number (else error)
            
  
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
                if mode == 'saving':
                    cv2.imwrite('VCS_Project/data/actual_thresholded/' + name, current_thresh)
                    return current_thresh
                else:
                    return current_thresh
                # changed : normally just return current_thresh, but need to generate some of the images
                # to check what we need for the synthetic dataset 
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                break
    
    
    def paint_image(self, img):
        drawing = False
        last_point = None
        img_display = img.copy()
        
        def on_brush_change(_):
            pass  # Just needed for trackbar creation
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, last_point
            if drawing:
                brush_size = cv2.getTrackbarPos('Brush Size', 'Paint')
                if last_point is not None:
                    cv2.line(img_display, last_point, (x, y), 0, brush_size*2)
                cv2.circle(img_display, (x, y), brush_size, 0, -1)
                last_point = (x, y)
                cv2.imshow('Paint', img_display)
            else:
                last_point = None
                
        cv2.namedWindow('Paint')
        cv2.createTrackbar('Brush Size', 'Paint', 10, 50, on_brush_change)
        cv2.setMouseCallback('Paint', mouse_callback)
        print("Hold 'd' to draw black, 'f' when finished, 'r' to reset")
        
        while True:
            cv2.imshow('Paint', img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('d'):
                drawing = True
            else:
                drawing = False
                
            if key == ord('r'):
                img_display = img.copy()
            elif key == ord('f'):
                cv2.destroyAllWindows()
                return img_display
            elif key == 27:
                cv2.destroyAllWindows()
                break
    
    
    def adaptive_dataset_generation(self, folder_path):
        # Used for generating the adaptive thresholded version of licenses plates of the actual dataset
        # and used to understand what we have to pay attention to in generation of noisy versions 
        # for the synthetic dataset
        path_vers = Path(folder_path)
        for path in path_vers.iterdir():
            if path.is_file():
                image_name = path.name.split('/')[-1]
        
            curr_path = folder_path + '/' + image_name
       
            # Get the image names that are currently in actual_thresholded
            actual_thresholded = os.listdir('VCS_Project/data/actual_thresholded')
            actual_thresholded = [img for img in actual_thresholded if img.endswith('.png')]
            
            

            # Check if the image is already thresholded
            if image_name in actual_thresholded:
                continue 
            
           
            else:
                img = self.scale_and_resize(cv2.imread(curr_path))
                img = self.perspective_correction(img=img)
                img = self.adaptive_thresholding(img=img,image_path= curr_path, mode='saving')

    def character_segmentation(self, image_path, deviation=10):
        image_path = image_path or self.IMAGE_PATH
        img_nmb = re.search(r'\d+', image_path).group()
        
        # Process image once
        img = self.scale_and_resize(cv2.imread(image_path))
        img = self.perspective_correction(img=img)
        img = self.adaptive_thresholding(img=img)
        img = self.paint_image(img)
        img = self.scale_and_resize(img)
        img_copy = img.copy()
        points = []
        counter = 0
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points, img_copy
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(img_copy, (x, y), 3, (255, 0, 0), -1)
                
                if len(points) == 4:
                    points_arr = np.array(points)
                    x_sorted = points_arr[np.argsort(points_arr[:, 0])]
                    left = x_sorted[:2]
                    right = x_sorted[2:]
                    left = left[np.argsort(left[:, 1])]
                    right = right[np.argsort(right[:, 1])]
                    
                    sorted_points = np.array([left[0], right[0], right[1], left[1]], dtype=np.int32)
                    cv2.polylines(img_copy, [sorted_points], True, (255, 0, 0), 2)
        
        cv2.namedWindow('Select Characters')
        cv2.setMouseCallback('Select Characters', mouse_callback)
        
        while True:
            cv2.imshow('Select Characters', img_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if len(points) == 4 and key == ord('f'):
                points_arr = np.array(points)
                x = min(points_arr[:, 0])
                y = min(points_arr[:, 1])
                w = max(points_arr[:, 0]) - x
                h = max(points_arr[:, 1]) - y
                
                roi = img[y:y+h, x:x+w]
                scale = min(20.0/w, 20.0/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                char_resized = cv2.resize(roi, (new_w, new_h))
                
                mnist_size = np.zeros((28, 28), dtype=np.uint8)
                x_offset = (28 - new_w) // 2
                y_offset = (28 - new_h) // 2
                mnist_size[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_resized
                
                os.makedirs(f"VCS_Project/results/license_plates/segmented_plate.{img_nmb}", exist_ok=True)
                cv2.imwrite(f"VCS_Project/results/license_plates/segmented_plate.{img_nmb}/character_{counter}.png", mnist_size)
                
                counter += 1
                points = []
                img_copy = img.copy()  # Reset image for next selection
                
            elif key == 27:  # ESC
                break
                
        cv2.destroyAllWindows()
        return None
    
    
    
    ############ Image processing methods ############
    
    @process
    def adjust_exposure_tool(self, factor: float = 1.0) -> str:
        """Adjusts the exposure of an image and saves it."""
        img = Image.open(self.current_image_path)
        enhanced_img = self.adjust_exposure(img, factor)
        enhanced_img.save(self.current_image_path)
    
    @process
    def adjust_brilliance_tool(self, factor: float = 1.0) -> str:
        """Adjusts the brilliance of an image and saves it."""
        img = Image.open(self.current_image_path)
        enhanced_img = self.adjust_brilliance(img, factor)
        enhanced_img.save(self.current_image_path)

    @process
    def adjust_contrast_tool(self, factor: float = 1.0) -> str:
        """Adjusts the contrast of an image and saves it."""
        img = Image.open(self.current_image_path)
        enhanced_img = self.adjust_contrast(img, factor)
        enhanced_img.save(self.current_image_path)

    @process
    def adjust_sharpness_tool(self, factor: float = 1.0) -> str:
        """Adjusts the sharpness of an image and saves it."""
        img = Image.open(self.current_image_path)
        enhanced_img = self.adjust_sharpness(img, factor)
        enhanced_img.save(self.current_image_path)

    @process
    def edge_detection_tool(self,) -> str:
        """Detects edges in an image and saves the result."""
        img = Image.open(self.current_image_path)
        edges = self.edge_detection(img)
        cv2.imwrite(self.current_image_path, edges)

    @process
    def grayscale_tool(self) -> str:
        """Converts an image to grayscale and saves it."""
        img = cv2.imread(self.current_image_path)
        processed_image = self.grayscale(img)
        cv2.imwrite(self.current_image_path, processed_image)    

    @process
    def invert_tool(self) -> str:
        """Inverts the colors of an image and saves it."""
        img = Image.open(self.current_image_path)
        processed_image = Image.eval(img, lambda x: 255 - x)
        processed_image.save(self.current_image_path)
        #cv2.imwrite(self.current_image_path, processed_image)  

    @process
    def auto_crop_tool(self):
        """Automatically crop the license plate from the image."""
        image = cv2.cvtColor(np.array(Image.open(self.current_image_path)), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        license_plate = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Look for a rectangle
                license_plate = approx
                break

        if license_plate is not None:
            # Create a mask for the plate and crop it
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [license_plate], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(license_plate)
            cropped = image[y:y + h, x:x + w]

            # Perspective transform to correct tilt
            pts = license_plate.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            processed_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            cv2.imwrite(self.current_image_path, processed_image)
        
        
      
    
    

    ############ Image workflow methods ############
    
    def update_step(self) -> str:
        """Update the step counter and save the image to a new path."""
        if self.step == 0:
            os.makedirs('process', exist_ok=True)
        self.step += 1
        self.max_steps = self.step
        
        # Save the image to the process folder
        img = Image.open(self.current_image_path)
        new_path = f'process/step_{self.step}.png'
        self.current_image_path = new_path
        img.save(new_path)
        return new_path
    
    def go_back(self, step = 1):
        """Go back to the previous step by loading the previous image."""
        if self.step - step > 1:
            self.step -= 1
            self.current_image_path = f'process/step_{self.step}.png'
        else:
            self.current_image_path = f'process/step_{self.step}.png'
            print("You have reached the first step.")
        return self.current_image_path

    def go_forward(self, step = 1):
        """Go forward to the next step by loading the next image."""
        if self.step + step < self.max_steps:
            self.step += 1
            self.current_image_path = f'process/step_{self.step}.png'
        else:
            self.current_image_path = f'process/step_{self.step}.png'
            print("You have reached the last step.")
        return self.current_image_path

    def restart(self):
        """Restart the agent by clearing the image path."""
        self.current_image_path = self.STARTING_IMAGE_PATH
        self.step = 0
        os.system("rm -rf process")
    
    
    
    
    
    ############ Image display methods ############
    
    def show_image(self):
        """Displays the image at the given path."""
        img = Image.open(self.current_image_path)
        img.show()
        
        




# Example usage
if __name__ == "__main__":
    SAMPLE_ID = 285
    IMAGE_PATH = f'results/license_plates/license_plate.{SAMPLE_ID}.png'
    #IMAGE_PATH = f'VCS_Project/results/license_plates/license_plate.{SAMPLE_ID}.png' # TODO : uncomment, need it bc I am working on a virtual environment and don't want do add it to git 

    #all_images = 'VCS_Project/results/license_plates'
    agent = LicensePlateAgent(image_path=IMAGE_PATH)

    #agent.adaptive_dataset_generation(all_images)
    #agent.character_segmentation(IMAGE_PATH)
    #agent.character_segmentation(IMAGE_PATH)
    #agent.adaptive_thresholding(IMAGE_PATH, block_size=111, constant=2, mode='processing')
    #agent.character_segmentation(f'VCS_Project/results/license_plates/license_plate.{SAMPLE_ID}_adaptive_thresholding.png')
    #agent.perspective_correction('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/results/license_plates/license_plate.2.png')
    


   
