#also create characters where some parts of the character are missing , like top part of a N or something like that for training (more robustness)

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
    
    imgs = []
    for contour in sorted(contours, key=lambda x: cv2.boundingRect(x)[0]):
        x, y, w, h = cv2.boundingRect(contour)
        # width to height ratio : the images of the characters will nearly always be highr than they are wide
        # w > 80 & h > 80 : we don't want to save the small contours, since they are probably noise

        #The characters on GERMAN licence plates, as well as the narrow rim framing it, are black on a white background.
        #  In standard size they are 75 mm (3 in) high, and 47.5 mm (1+7⁄8 in) wide for letters or 44.5 mm (1+3⁄4 in) wide for digits
        # 47.5/75 = 0.6333, 44.5/75 = 0.5933
     #   print(f'w/h : {w/h}')
        if 0.57 < w/h < 0.65:
            reg_of_interest = image[y:y+h, x:x+w] # region of interest : the rectangle area that we found ; also take it from the original image!

         #   print(f'shape of reg_of_interest : {reg_of_interest.shape}')



            
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
            imgs.append(mnist_size)


    
   # assert len(imgs) == 1, f"Expected 1 character, got {len(imgs)}" # TODO : sometimes 2 bc it returns the same one .... need some way to fix this for generation process
    cv2.imwrite('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/data/synthetic/german_font/clean/B/B.png', imgs[0])
    return imgs[0]


def create_augmented_pair(clean_image):
    """Create a noisy version of the input image"""
    # Convert PIL to numpy array
    img_array = np.array(clean_image)
    
    # Define augmentation pipeline
    # TODO : research what is best here, we need to get it as close as possible to the real world 
    augmenter = iaa.Sequential([
        iaa.Sometimes(0.7, iaa.GaussianBlur(sigma=(0.5, 2.0))),
        iaa.Sometimes(0.6, iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255))),
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        iaa.Sometimes(0.3, iaa.MotionBlur(k=(3, 7)))
    ])
    
    # Apply augmentation
    noisy_image = augmenter(image=img_array)
    return Image.fromarray(noisy_image)


def create_directories(output_dir):
    # Define character set (German license plates)
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    os.makedirs(output_dir + '/clean')
    os.makedirs(output_dir + '/noisy')
    for char in chars:
        os.makedirs(output_dir + '/clean/' + char)
        os.makedirs(output_dir + '/noisy/' + char)


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
        clean_image.save(clean_dir + f'/{char}/' + f'{char}.png')

        for i in range(100):  # Generate 100 augmented versions of each character
            # Generate corresponding noisy image
            noisy_image = create_augmented_pair(clean_image)
            # Save images
            noisy_image.save(noisy_dir + f'/{char}/' + f'{char}_{i}.png')


def adaptive_thresholding(folder_path,block_size = 111):
    """Needed to preprocess our synthetic data such that it will have the same format as the real data
       that we will try to denoise.
       
    """

    img = cv2.imread(folder_path + f'/{folder_path[-2]}.png')

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Take L channel (lightness)
    # This helps a lot, since the characters on the (european) license plates are usually black, thus have a low lightness
    # that can be used to separate them from the background, even in blurry images
    l_channel = lab[:,:,0]


    # Apply adaptive thresholding based on lightness
    thresh = cv2.adaptiveThreshold(
        l_channel,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,  # block size 
        1   # constant subtracted from mean
    )

    cv2.imwrite(folder_path + f'/{folder_path[-2]}_adaptive.png', thresh)
    

  #  return thresh 


def resizing(folder_path):
    """Resize images in the style of MNIST : 20x20 for the character itself, 28x28 for the whole image"""
    # Load image
    img = cv2.imread(folder_path + f'/{folder_path[-2]}_adaptive.png', cv2.IMREAD_GRAYSCALE)
    
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Crop image to bounding box
    cropped_img = img[y:y+h, x:x+w]
    
    # Resize to 20x20
    resized_img = cv2.resize(cropped_img, (20, 20))
    
    # Create blank 28x28 image
    final_img = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate position to center the character
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    
    # Insert character in the center of the blank image
    final_img[y_offset:y_offset+20, x_offset:x_offset+20] = resized_img
    
    # Save final image
    cv2.imwrite(folder_path + f'/{folder_path[-2]}_resized.png', final_img)





    



#generate_single_character('B','/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/fonts/FE-FONT.TTF')

#create_augmented_pair(img_A).show()
create_directories('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/data/synthetic/german_font')

#generate_dataset('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/data/synthetic/german_font','/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/fonts/FE-FONT.TTF')

#adaptive_thresholding('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/data/synthetic/german_font/clean/0/', '0')
#cv2.waitKey(0)
#cv2.destroyAllWindows()




