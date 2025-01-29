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
def generate_single_character(char, font_path, image_size=(100, 150)):
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
    LETTER_RATIO = 47.5 / 75  # ≈ 0.6333
    DIGIT_RATIO = 44.5 / 75   # ≈ 0.5933
    
    # Determine if character is a letter or digit and set target ratio
    target_ratio = LETTER_RATIO if char in string.ascii_uppercase else DIGIT_RATIO
    
    # Calculate target dimensions within the image
    image_ratio = image_size[0] / image_size[1]
    
    # CHANGED: Increased the scaling factor from 90% to 95% to utilize more of the canvas
    if image_ratio > target_ratio:  # Height determines size
        target_height = int(image_size[1] * 0.95)  # 95% of image height
        target_width = int(target_height * target_ratio)
    else:  # Width determines size
        target_width = int(image_size[0] * 0.95)  # 95% of image width
        target_height = int(target_width / target_ratio)
    
    # Binary search to find font size
    min_size = 1
    max_size = max(image_size) * 2
    optimal_size = None
    optimal_bbox = None
    
    while min_size <= max_size:
        current_size = (min_size + max_size) // 2
        font = ImageFont.truetype(font_path, size=current_size)
        bbox = draw.textbbox((0, 0), char, font=font)
        
        current_width = bbox[2] - bbox[0]
        current_height = bbox[3] - bbox[1]
        
        # CHANGED: Adjusted to ensure the optimal size respects the new target dimensions
        if (abs(current_width - target_width) <= 1 and
            abs(current_height - target_height) <= 1):
            optimal_size = current_size
            optimal_bbox = bbox
            break
        elif (current_width > target_width or 
              current_height > target_height):
            max_size = current_size - 1
        else:
            min_size = current_size + 1
            if optimal_size is None or current_width > target_width * 0.8:
                optimal_size = current_size
                optimal_bbox = bbox
    
    # Use the found optimal font size
    font = ImageFont.truetype(font_path, size=optimal_size)
    bbox = optimal_bbox if optimal_bbox else draw.textbbox((0, 0), char, font=font)
    
    # Center the character
    x = (image_size[0] - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (image_size[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
    
    # Draw the character
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



# TODO : remove some/add some, make pictures not that fucked up !!!
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
            iaa.Affine(scale=(0.8, 1.2)) # TODO : fix 
        ])),
        
        
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
        

           
    ])
    
    # Apply augmentation
    noisy_image = augmenter(image=image)
    
    return Image.fromarray(noisy_image)




def create_augmented_pair_edge_roughness(clean_image):
    """Create noisy versions based on edge roughness"""

    # TODO : Preprocessing with contour finding doesn't work if image is too fizzy (i.e. right now values -2 and 2, already doesn't work)
    image = np.array(clean_image)

    # in cv2, for contours background should be black and object should be white
    
    image = cv2.bitwise_not(image)


    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_areas = [(contour,cv2.contourArea(contour)) for contour in contours]

    # Sort contour areas by the area (from biggest to smallest)
    contour_areas = sorted(contour_areas, key=lambda x: x[1], reverse=True)
    
    # Now, the idea is : 
    # First draw in the biggest contour in the color of the object (white) (this will overdraw the wholes in numbers etc.)
    # But then draw in the smaller contours in black again
    for idx, contours in enumerate(contour_areas):
        if idx == 0: # biggest area
            cv2.drawContours(image, [contours[0] + np.random.randint(-2, 2, contours[0].shape)], -1, 255, -1)
        else: # smaller areas
            cv2.drawContours(image, [contours[0] + np.random.randint(-2,2, contours[0].shape)], -1, 0, -1)
      

    cv2.imshow('edge_roughness', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

    # Invert again (bc with current logic, preprocessing expects white background and black object)
    image = cv2.bitwise_not(image)
    


    return Image.fromarray(image)


def create_augmented_pair_thickness(clean_image, area = (5,8)):
    """
    Erosion & Dilation to simulate thickness variations.
    https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    """

    image = np.array(clean_image)

    kernel_size = random.randint(area[0], area[1])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if random.random() > 0.5:
        image = cv2.erode(image,kernel)
        cv2.imshow('erode', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
       
        return image
    
    image = cv2.dilate(image, kernel)
    cv2.imshow('dilate', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
    return image




def create_augmented_pair_perspective_skew(clean_image):

    image = np.array(clean_image)

    # Just needed here, since fit_output resizes image to original size and does a black background automatically,
    # so like this we can just use the image as it is
    image = cv2.bitwise_not(image)

    # Perspective variations
    augmenter = iaa.PerspectiveTransform(scale=(0.05, 0.2), keep_size=True, fit_output=True)


    image = augmenter(image=image)

    # Invert again for preprocessing logic later on
    image = cv2.bitwise_not(image)


    cv2.imshow('perspective_correction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   # exit()

    # TODO : add perspective correction again ??? bc with actual license plates we correct always, but I don't know if that makes sense here like that or not
    # Or maybe even try to train on skewed perspectives, don't know if vAE can handle it, but maybe even sometimes letting it skewed can improve readability, 
    # atleast for humans ; we have to test 
    # TODO : or just instead of that, we can add that characters are pushed more together bc I think that is what happens when we correct the perspective ;
    # so make characters appear more tight together, for a P/A etc holes more small etc. 
    return Image.fromarray(image)

    

def create_augmented_pair_imagequality(clean_image):
    """Create noisy versions based on image quality"""

    image = np.array(clean_image)

    # I took out gaussian noise, bc in the real license plate image, with adaptive thresholding we can remove noise or then with the painting tool,
    # thus we will not have like noisy dots around the image anyway
    # Image quality
    augmenter = iaa.JpegCompression(compression=(70, 90)) 


    image = augmenter(image=image)

    cv2.imshow('image_quality', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

    return Image.fromarray(image)


def create_augemented_pair_character_degradation(clean_image):
    # TODO : never this extreme in images, but maybe helps a bit ?
    """Create noisy versions based on character degradation"""
    
    image = np.array(clean_image)

    # Sets pixels in image to zero per chance, i.e. the character must be white and the background black
    image = cv2.bitwise_not(image)

    augmenter = iaa.Dropout(p=(0.02,0.05))

    image = augmenter(image=image)

    # Invert again for preprocessing logic later on
    image = cv2.bitwise_not(image)

    cv2.imshow('character_degradation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

    return Image.fromarray(image)



def create_augmented_pair_occlusion(clean_image):
    """Create noisy versions based on occlusion"""
    pass

def create_augmented_pair_blur(clean_image):
    """Create noisy versions based on blur"""
    pass









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


    # New logic : 
    # 1. We know that the character is white and the background is black, i.e. pixel values are 0 for the background and >0 for the character
    # 2. To find a rectangle around the character, we just look at all pixels that are != 0 and find the bounding rectangle around them
    # That means : Find the white pixels that is most left in regard to all white pixels, most right, most top and most bottom
    # Use these to gather the 4 corner points of the rectangle and then crop the image to this rectangle
    # Then do the normal resizing steps

    # Find all white pixels
    white_pixels = np.where(image > 0)

    # Get the bounds
    min_y = np.min(white_pixels[0])  # top
    max_y = np.max(white_pixels[0])  # bottom
    min_x = np.min(white_pixels[1])  # left
    max_x = np.max(white_pixels[1])  # right


    reg_of_interest = image[min_y:max_y, min_x:max_x] 

    # Get width and height of region of interest
    h, w = reg_of_interest.shape

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

   # cv2.imshow('correct_img', correct_img)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
  

    return Image.fromarray(correct_img)



    '''
    # Apply canny edge detection, for contours 
    canny = cv2.Canny(image, 100, 200)




    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


   # Get all the areas of the contours
    areas = [cv2.contourArea(contour) for contour in contours]
    # Get the max area (since we have black surrounding and white character, the character will have the biggest area)
    max_area = max(areas)


    correct_img = None 
    for contour in sorted(contours, key=lambda x: cv2.boundingRect(x)[0]):
        x, y, w, h = cv2.boundingRect(contour)
   
        area = cv2.contourArea(contour)
        if area == max_area:


            reg_of_interest = image[y:y+h, x:x+w] # region of interest : the rectangle area that we found

         



   
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

        
    cv2.imshow('correct_img', correct_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   # exit()

    return Image.fromarray(correct_img)
    '''

import random 
def generate_dataset_tester(output_dir, font_path):
    """Used to test the single augmentation methods"""
    # Define character set (German license plates)
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    clean_dir = output_dir + '/clean'
    noisy_dir = output_dir + '/noisy'
    
    # Generate multiple versions of each character
    for char in chars:

        clean_image = generate_single_character(char, font_path)


        for i in range(1):  

            noisy_image_processed = None 
            while noisy_image_processed == None: 
                # Generate corresponding noisy image
                noisy_image = create_augemented_pair_character_degradation(clean_image)
                # Preprocess image
                noisy_image_processed = preprocessing(noisy_image)
            
            noisy_image_processed.save(noisy_dir + f'/{char}/' + f'{char}_{i}.png')




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
        clean_image_processed.save(clean_dir + f'/{char}/' + f'{char}_original.png')

        for i in range(100):  # Generate 100 augmented versions of each character
            # Generate clean variations
            clean_variation = create_clean_variations(clean_image)
            # Preprocess image
            clean_variation_processed = preprocessing(clean_variation)
            # Save images
            clean_variation_processed.save(clean_dir + f'/{char}/' + f'{char}_{i}.png')

            # None logic needed bc sometimes the preprocessing function doesn't find any contours or gets some error ; in that case
            # we just retry until we get no errors anymore 
            noisy_image_processed = None 
            while noisy_image_processed == None: 
                # Generate corresponding noisy image
                noisy_image = create_augmented_pair(clean_image)
                # Preprocess image
                noisy_image_processed = preprocessing(noisy_image)
                # Additionally, we will do some checks to make sure our images are not too noisy!
                # We saw both : images that basically are just a straight line for E, making it impossible to recognize it and
                # images that have like just a white square in the middle, which is also not helpful
                # Therefore, we try to remove these
                clean_image_np = np.array(clean_image_processed)
                noisy_image_np = np.array(noisy_image_processed)
                black_pixels_clean = np.sum(clean_image_np == 0) # 0,1,2,3,4,5,6...
                black_pixels_noisy = np.sum(noisy_image_np == 0)
                ratio = black_pixels_noisy/black_pixels_clean

                if ratio < 0.8 or ratio > 1.1: # hardcoded for now, also still some problematic images can get through
                    noisy_image_processed = None 
                    print(f'Image {char}_{i} too noisy, retrying...')

                # TODO : create a classification NN that classifies whether an image is too mucb noise (so trained on clean images & noisy images binary classification (i.e. sigmoid threshold))
                # such that NN is called here every time and helps us in not having too noisy training examples later 



            # Save images
            noisy_image_processed.save(noisy_dir + f'/{char}/' + f'{char}_{i}.png')







#generate_dataset('data/synthetic/german_font','fonts/FE-FONT.TTF')
generate_dataset_tester('VCS_Project/data/synthetic/german_font','VCS_Project/fonts/FE-FONT.TTF')



