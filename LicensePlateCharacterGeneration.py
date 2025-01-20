#also create characters where some parts of the character are missing , like top part of a N or something like that for training (more robustness)


# !!! imgaug must be downloaded from this fork : https://github.com/marcown/imgaug

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import imgaug.augmenters as iaa
import os 
import cv2
import string


# image_size should be the same as the size of the characters we segment from the actual license plates BEFORE resizing them to 28x28 ! Such that
# we do the same transformations !!!
# TODO : maybe do some changes here, img_size 90,140 is not always diretly the one to use but its around the right area (atleast from when I segmented the characters BEFORE
# resizing them to 28x28, they had about this size 
# TODO : I didn't manage for the letters/numbers to fill out the complete 90,140 , like it should be the case 
def generate_single_character(char, font_path, image_size=(100,150)):
    """
    Creates an image of a character following German license plate standards.
    Characters maintain exact width-to-height ratios:
    - Letters: 47.5/75 = 0.6333
    - Digits: 44.5/75 = 0.5933
    
    Args:
        char (str): Single character to render (letter or digit)
        font_path (str): Path to the FE-Schrift font file
        image_size (tuple): Size of the output image (width, height)
        
    Returns:
        PIL.Image: The generated image
    """
    # Create blank image with white background
    image = Image.new('L', image_size, color='white')
    draw = ImageDraw.Draw(image)
    
    # Define target ratios based on German standards
    LETTER_RATIO = 47.5/75  # ≈ 0.6333
    DIGIT_RATIO = 44.5/75   # ≈ 0.5933
    
    # Determine if character is a letter or digit and set target ratio
    target_ratio = LETTER_RATIO if char in string.ascii_uppercase else DIGIT_RATIO
    
    # Calculate target dimensions within the image
    image_ratio = image_size[0] / image_size[1]
    
    # If image is wider than character should be, height determines size
    if image_ratio > target_ratio:
        target_height = int(image_size[1] * 0.9)  # 90% of image height
        target_width = int(target_height * target_ratio)
    # If image is narrower than character should be, width determines size
    else:
        target_width = int(image_size[0] * 0.9)  # 90% of image width
        target_height = int(target_width / target_ratio)
    
    # Binary search to find font size that gives correct dimensions
    min_size = 1
    max_size = max(image_size) * 2  # Start with a large enough maximum
    optimal_size = None
    optimal_bbox = None
    
    while min_size <= max_size:
        current_size = (min_size + max_size) // 2
        font = ImageFont.truetype(font_path, size=current_size)
        bbox = draw.textbbox((0, 0), char, font=font)
        
        current_width = bbox[2] - bbox[0]
        current_height = bbox[3] - bbox[1]
        
        # Check if dimensions are within 1 pixel of target
        if (abs(current_width - target_width) <= 1 and 
            abs(current_height - target_height) <= 1):
            optimal_size = current_size
            optimal_bbox = bbox
            break
        # If too big, decrease size
        elif (current_width > target_width or 
              current_height > target_height):
            max_size = current_size - 1
        # If too small, increase size
        else:
            min_size = current_size + 1
            # Keep track of best size so far
            if optimal_size is None or current_width > target_width * 0.8:
                optimal_size = current_size
                optimal_bbox = bbox
    
    # Use the found optimal font size
    font = ImageFont.truetype(font_path, size=optimal_size)
    bbox = optimal_bbox if optimal_bbox else draw.textbbox((0, 0), char, font=font)
    
    # Center the character
    x = (image_size[0] - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (image_size[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
    
    # Draw the character in black
    draw.text((x, y), char, font=font, fill='black')


    return image





def create_clean_variations(clean_image):
    """Create slightly varied but still clean versions of the input image
    
    Args:
        clean_image: PIL Image or np.array of the clean character
        
    Returns:
        PIL Image of the slightly augmented version
    """
    # Convert to numpy array if not already
    image = np.array(clean_image, dtype=np.uint8)
    
    # Define very mild augmentation pipeline
    augmenter = iaa.Sequential([
        # Slight perspective changes
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.02))),
        
        # Minimal rotation
        iaa.Sometimes(0.5, iaa.Rotate((-2, 2))),
        
        # Very slight scaling
        iaa.Sometimes(0.5, iaa.Affine(
            scale=(0.95, 1.05),
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}
        )),
        
        # Subtle thickness variations using elastic transformation
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0.1, 0.5), sigma=(0.05, 0.1))),
        
        # Very mild brightness adjustment
        iaa.Sometimes(0.3, iaa.Add((-10, 10)))
    ])

    # Apply augmentation
    clean_variation = augmenter(image=image)
    
    return Image.fromarray(clean_variation)


def create_augmented_pair(clean_image):
    """Create a noisy version of the input image that simulates real-world license plate conditions
    
    Args:
        clean_image: PIL Image or np.array of the clean image
        
    Returns:
        PIL Image of the augmented version
    """
    # Convert to numpy array if not already
    image = np.array(clean_image)

    
    # Define augmentation pipeline
    augmenter = iaa.Sequential([
        # Camera and motion effects
        iaa.Sometimes(0.8, iaa.OneOf([
            # Motion blur for vehicle movement
            iaa.MotionBlur(k=(7, 15), angle=(-45, 45)),
            # Camera shake and focus issues
            iaa.GaussianBlur(sigma=(0.5, 3.0)),
            # Defocus blur
            iaa.AverageBlur(k=(2, 5))
        ])),
        
        # Perspective and distance variations
        iaa.Sometimes(0.7, iaa.Sequential([
            # More aggressive perspective changes
            iaa.PerspectiveTransform(scale=(0.05, 0.15)),
            # Slight rotations
          #  iaa.Rotate((-5, 5)),
            # Distance variations
            iaa.Affine(scale=(0.8, 1.2))
        ])),
        
        # Lighting and exposure effects
        # I think they fuck up the picture too much, also don't know if it really makes sense on black/white images
       # iaa.Sometimes(0.8, iaa.OneOf([
            # Contrast changes
       #     iaa.LinearContrast((0.6, 1.4)),
            # Uneven lighting
       #     iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.3, 0.7))
       # ])),
        
        # Image quality and noise
        iaa.Sometimes(0.7, iaa.OneOf([
            # Camera sensor noise
            iaa.AdditiveGaussianNoise(scale=(0, 0.15 * 255)),
            # Compression artifacts
            iaa.JpegCompression(compression=(70, 90)),
        ])),
        
        # Character degradation and occlusion
        iaa.Sometimes(0.6, iaa.OneOf([
            # Small occlusions
            iaa.Dropout(p=(0.01, 0.1)),
            # Larger occlusions
            iaa.CoarseDropout(p=(0.02, 0.15), size_percent=(0.02, 0.05))
        ])),
        

        # this is problematic bc it then doesn't find any contours sometimes
       
      #  # Weather effects (use sparingly)
     #   iaa.Sometimes(0.3, iaa.OneOf([
            # Light rain effect
        #    iaa.RainLayer(
        #        density=(0.1, 0.2),
        #        density_uniformity=0.2,
        #        drop_size=(0.01, 0.02),
        #        drop_size_uniformity=0.1,
        #        angle=(-15, 15),
        #        speed=(0.1, 0.2),
        ##        blur_sigma_fraction=(0.001, 0.001)
         #   ),
            # Light fog
        #    iaa.Fog()
       # ])),
           
    ])
    
    # Apply augmentation
    noisy_image = augmenter(image=image)
    
    return Image.fromarray(noisy_image)


'''
def create_augmented_pair(clean_image):
    """Create a noisy version of the input image
       clean_image : np.array of the clean image"""

    # Convert to numpy array
    image = np.array(clean_image)

    # Define augmentation pipeline
    # TODO : research what is best here, we need to get it as close as possible to the real world 
    augmenter = iaa.Sequential([
        iaa.Sometimes(0.7, iaa.GaussianBlur(sigma=(0.5, 2.0))),
        iaa.Sometimes(0.6, iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255))),
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        iaa.Sometimes(0.3, iaa.MotionBlur(k=(3, 7)))
    ])
    
    # Apply augmentation
    noisy_image = augmenter(image=image)
    return Image.fromarray(noisy_image)
'''

def create_directories(output_dir):
    # Define character set (German license plates)
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    os.makedirs(output_dir + '/clean')
    os.makedirs(output_dir + '/noisy')
    for char in chars:
        os.makedirs(output_dir + '/clean/' + char)
        os.makedirs(output_dir + '/noisy/' + char)


def preprocessing(image):

    image = np.array(image)


    # invert colors
    image = cv2.bitwise_not(image)


    # Apply canny edge detection, for contours 
    canny = cv2.Canny(image, 100, 200)


    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   # cv2.drawContours(image, contours, -1, 128, 2)
   # cv2.imshow('img', image)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()

   # Get all the areas of the contours
    areas = [cv2.contourArea(contour) for contour in contours]
    # Get the max area (since we have black surrounding and white character, the character will have the biggest area)
    max_area = max(areas)

    
    correct_img = None 
    for contour in sorted(contours, key=lambda x: cv2.boundingRect(x)[0]):
        x, y, w, h = cv2.boundingRect(contour)
        # width to height ratio : the images of the characters will nearly always be highr than they are wide
        # w > 80 & h > 80 : we don't want to save the small contours, since they are probably noise

        #The characters on GERMAN licence plates, as well as the narrow rim framing it, are black on a white background.
        #  In standard size they are 75 mm (3 in) high, and 47.5 mm (1+7⁄8 in) wide for letters or 44.5 mm (1+3⁄4 in) wide for digits
        # 47.5/75 = 0.6333, 44.5/75 = 0.5933


        # We know that the character is centered, thus the middle of the rectangle that stems from the contour of the character
        #  should have coordinates (50,75) (as the image is 100x150 in total size)
   
        area = cv2.contourArea(contour)
        if area == max_area:
            reg_of_interest = image[y:y+h, x:x+w] # region of interest : the rectangle area that we found ; also take it from the original image!

   
            # Calculate scaling factor to fit in 20x20 box while maintaining aspect ratio
            scale = min(20.0/w, 20.0/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize character to fit in 20x20 box
            try:
                char_resized = cv2.resize(reg_of_interest, (new_w, new_h))
            except cv2.error as e:
                print('Error resizing the image ; skipping this image :', e)
                return None 
            
            # Create 28x28 blank (black) image
            mnist_size = np.zeros((28, 28), dtype=np.uint8)
            
            # Calculate position to center character
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            
            # Place character in center of 28x28 image
            mnist_size[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_resized
            correct_img = mnist_size


    return Image.fromarray(correct_img)


def generate_dataset(output_dir, font_path):
    """Generate dataset of clean and noisy character images"""
    # Define character set (German license plates)
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    clean_dir = output_dir + '/clean'
    noisy_dir = output_dir + '/noisy'
    
    # Generate multiple versions of each character
    for char in chars:

        # Generate clean image (just one since it's always the same ; TODO : possibly in different sizes could help, since license plate may be from diff perspectives?)
        clean_image = generate_single_character(char, font_path)
        # Preprocess image
        clean_image_processed = preprocessing(clean_image)
        clean_image_processed.save(clean_dir + f'/{char}/' + f'{char}.png')

        for i in range(100):  # Generate 100 augmented versions of each character
            # Generate clean variations
            clean_variation = create_clean_variations(clean_image)
            # Preprocess image
            clean_variation_processed = preprocessing(clean_variation)
            # Save images
            clean_variation_processed.save(clean_dir + f'/{char}/' + f'{char}_{i}.png')

            # Generate corresponding noisy image
            noisy_image = create_augmented_pair(clean_image)
            # Preprocess image
            noisy_image_processed = preprocessing(noisy_image)
            # Save images
            if noisy_image_processed is None:
                continue
            else:
                noisy_image_processed.save(noisy_dir + f'/{char}/' + f'{char}_{i}.png')










#img_b = generate_single_character('B','/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/fonts/FE-FONT.TTF')

#create_augmented_pair(img_A).show()
#create_directories('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/data/synthetic/german_font')

generate_dataset('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/data/synthetic/german_font','/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/fonts/FE-FONT.TTF')

#adaptive_thresholding('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/data/synthetic/german_font/clean/0/', '0')
#cv2.waitKey(0)
#cv2.destroyAllWindows()




