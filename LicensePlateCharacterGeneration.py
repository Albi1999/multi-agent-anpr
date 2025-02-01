# !!! imgaug must be downloaded from this fork : https://github.com/marcown/imgaug

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import imgaug.augmenters as iaa
import os 
import cv2
import string
import random 
import itertools
from pathlib import Path



# image_size should be the same as the size of the characters we segment from the actual license plates BEFORE resizing them to 28x28 ! Such that
# we do the same transformations
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


def create_augmented_pair_edge_roughness(clean_image, strength_roughness_choice = (6,7,8,9,10,11)):
    """Create noisy versions based on edge roughness"""

    strength_roughness = random.choice(strength_roughness_choice)

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
            cv2.drawContours(image, [contours[0] + np.random.randint(-strength_roughness, strength_roughness, contours[0].shape)], -1, 255, -1)
        else: # smaller areas
            cv2.drawContours(image, [contours[0] + np.random.randint(-strength_roughness,strength_roughness, contours[0].shape)], -1, 0, -1)
      

    # Invert again (bc with current logic, preprocessing expects white background and black object)
    image = cv2.bitwise_not(image)
    

    return Image.fromarray(image)


def create_augmented_pair_thickness(clean_image, area = (5,8)):
    """
    Erosion & Dilation to simulate thickness variations.
    https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    """

    image = np.array(clean_image)

    # Since dilation & erosion work for white characters on black background
    image = cv2.bitwise_not(image)

    kernel_size = random.randint(area[0], area[1])

    # For the erosion, we don't want to do too much, because most of the characters found in license plates are rather thick
    kernel_erode = np.ones((2,2), np.uint8)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
  #  if random.random() > 0.5:
  #      image = cv2.erode(image,kernel_erode)
       
  #      return image
    
    image = cv2.dilate(image, kernel)

    image = cv2.bitwise_not(image)
  
    return image


def create_augmented_pair_perspective_skew(clean_image, scale=(0.05, 0.2)):

    image = np.array(clean_image)

    # Just needed here, since fit_output resizes image to original size and does a black background automatically,
    # so like this we can just use the image as it is
    image = cv2.bitwise_not(image)

    # Perspective variations
    augmenter = iaa.PerspectiveTransform(scale=scale, keep_size=True, fit_output=True)

    image = augmenter(image=image)

    # Invert again for preprocessing logic later on
    image = cv2.bitwise_not(image)


    return Image.fromarray(image)

    
def create_augmented_pair_imagequality(clean_image, compression_strength = (70, 90)):
    """Create noisy versions based on image quality"""

    image = np.array(clean_image)

    # I took out gaussian noise, bc in the real license plate image, with adaptive thresholding we can remove noise or then with the painting tool,
    # thus we will not have like noisy dots around the image anyway
    # Image quality
    augmenter = iaa.JpegCompression(compression= compression_strength) 

    image = augmenter(image=image)

    return Image.fromarray(image)


def get_edge_segment(canny_image, start_point, segment_length=5):
    """Helper function for create_augmented_pair_occlusion"""
    height, width = canny_image.shape
    y, x = start_point
    segment = [(y, x)]
    
    # Define 8-connected neighborhood (such that we have movement in all directions from the current pixel)
    neighbors = [(-1,-1), (-1,0), (-1,1),
                (0,-1),         (0,1),
                (1,-1),  (1,0),  (1,1)]
    
    # Follow edge in both directions
    for direction in [1, -1]:  # Forward and backward (the idea is that if it already moved in one direction, these pixels will be in segment and thus in the next iteration, it will move in the other direction)
        current_y, current_x = y, x
        steps = 0
        
        while steps < segment_length:
            # Look for next edge pixel in neighborhood
            found_next = False
            for dy, dx in neighbors:
                ny, nx = current_y + dy, current_x + dx
                
                # Check bounds and if it's an edge pixel
                if (0 <= ny < height and 0 <= nx < width and 
                    canny_image[ny, nx] > 0 and 
                    (ny, nx) not in segment):
                    segment.append((ny, nx))
                    current_y, current_x = ny, nx
                    found_next = True
                    break
            
            if not found_next:
                break  # No more connected edge pixels
                
            steps += 1
    
    return segment


def remove_pixels_segment(image, edge_segment, length_max = 30):
    height, width = image.shape
    result = image.copy()  
    
    neighbors = [(-1,-1), (-1,0), (-1,1),
                (0,-1),         (0,1),
                (1,-1),  (1,0),  (1,1)]

 
    already_removed = set(edge_segment)
    

    for idx, (y, x) in enumerate(edge_segment):
        current_y, current_x = y, x
        result[current_y, current_x] = 0
        
        removal_length = np.random.randint(1, length_max)
        steps = 0

        while steps < removal_length:
            # Shuffle neighbors to avoid bias in direction
            random_neighbors = np.random.permutation(neighbors)
            found_next = False
            
            for dy, dx in random_neighbors:
                ny, nx = current_y + dy, current_x + dx

                if (0 <= ny < height and 0 <= nx < width and 
                    result[ny, nx] > 0 and 
                    (ny, nx) not in already_removed):
                    
                    result[ny, nx] = 0
                    already_removed.add((ny, nx))  
                    current_y, current_x = ny, nx
                    steps += 1
                    found_next = True
                    break
            
            if not found_next:
                break  # No valid pixels found, stop this removal sequence

    return result


def create_augmented_pair_occlusion(clean_image, n_segments_choice = (7,8,9,10,11,12), segment_length_choice = (7,8,9,10,11,12), length_max_choice = tuple(np.arange(150, 200, 1))):
    """Create noisy versions based on occlusion"""
    # Which pixels ? --> we don't want to remove the ones "inside" of the character,
    # but rather work around the edges and remove parts there, at some areas more than others
    # Therefore, usy Canny Edge Detector first, and around these edges, we do some erosion 

    n_segments = random.choice(n_segments_choice)
    segment_length = random.choice(segment_length_choice)
    length_max = random.choice(length_max_choice)

    image = np.array(clean_image)

    # Since canny stores edges as white, might aswell make the character now white for consistency (and background black)
    image = cv2.bitwise_not(image)


    # Use canny edge detection to find edges
    canny = cv2.Canny(image, 100, 200)

    # canny stores the edge locations as white pixels 
    # Now we randomly select some white pixel and its surrounding area (area size based on some parameter)
    # and look into all directions to decide where in the original image the white areas are and then "move into"
    # the character to remove some parts of it 

    # Get the y & x coordinates for all white pixels (i.e. edges, since we used canny)
    edges_y , edges_x = np.where(canny > 0)

    used_segments = set()
    for _ in range(n_segments):

        cond = True
        while cond:
            # Randomly select a starting point
            rand_idx = np.random.randint(0, len(edges_y))
            start_point = (edges_y[rand_idx], edges_x[rand_idx])

            # Check if start_point was already used (i.e used in some segment)
            if start_point not in used_segments:
                cond = False 

        edge_segment = get_edge_segment(canny, start_point, segment_length=segment_length)

        image = remove_pixels_segment(image, edge_segment, length_max=length_max)




        # Add the segment to the used segments
        used_segments.update(edge_segment)

    # Invert again for preprocessing logic later on
    result_image = cv2.bitwise_not(image)


    return Image.fromarray(result_image)


def create_augmented_pair_hole_filling(clean_image):
    # Note that this doesn't work on all characters, because not all holes are completely "closed"

    # TODO : fix the bitwise_not madness
    image = np.array(clean_image)
    # For dialation & erosion operations : we need black background and white characters
    image = cv2.bitwise_not(image)

    # First, dilate the image (this is in order to close some of the holes, such that the flooding function will work better and on more characters)
    kernel_dil = np.ones((8,8), np.uint8)  # Adjust kernel size to control how much to dilate
    dilated = cv2.dilate(image, kernel_dil, iterations=1)  # Adjust iterations for more/less dilation
    image = dilated
    image = cv2.bitwise_not(image)
    height, width = image.shape
    mask = np.zeros((height + 2, width + 2), np.uint8)
    
    # Clone the image for flood filling
    fill_img = image.copy()
    
    # Flood fill from point (0,0)
    cv2.floodFill(fill_img, mask, (0,0), 0)
    
    # Invert to get holes
    holes = cv2.bitwise_not(fill_img)

    # Note that the holes are now in black, so we need to invert the image again, since erosion works on white characters
    holes = cv2.bitwise_not(holes)
    
    kernel_eros = np.ones((11,11), np.uint8)  # Adjust kernel size to control how much to erode
    eroded_holes = cv2.erode(holes, kernel_eros, iterations=1)  # Adjust iterations for more/less erosion

    # Note that we want to close the holes, therefore we need to get the pixel difference between the holes and the eroded holes
    # because that will be the "outer" part that we want to keep
    diff = cv2.absdiff(holes, eroded_holes)

   # exit()
    # Combine with original image
    # First, invert again 
    eroded_hole_final = cv2.bitwise_not(diff)
    filled = cv2.bitwise_and(image, eroded_hole_final)

    # Finally, erode the image to try to get more back to normal; use the same kernel as in dilation
    filled = cv2.bitwise_not(filled)
    eroded = cv2.erode(filled, kernel_dil, iterations=1)
    eroded = cv2.bitwise_not(eroded)

    return Image.fromarray(eroded)


def create_augmented_pair_smushing(clean_image, factor_choice=(0.4,0.5,0.6)):
    # After initial testing, we found that a factor of 0.3 was too much and made characters too unrecognizable,
    # thus we switched to 0.6
    # For stretching : factor > 1, for smushing : factor < 1    
    image = np.array(clean_image)
    height, width = image.shape
    factor = random.choice(factor_choice)
    compressed_width = int(width * factor)
    
    # Resize using cv2
    compressed_img = cv2.resize(image, (compressed_width, height), 
                              interpolation=cv2.INTER_LINEAR)


    return Image.fromarray(compressed_img)


def create_augmented_pair_rotation(clean_image, rotation_strength = (-7,7)):
    # Rotate image 
    image = np.array(clean_image)

    # this is simply since rotation will rotate whole image and then we get little black dots in the edges after rotation, with this
    # we can in the end then just invert again 
    image = cv2.bitwise_not(image)

    augmenter = iaa.Rotate(rotate= rotation_strength)

    image = augmenter(image=image)

    # Invert again 
    image = cv2.bitwise_not(image)

    return Image.fromarray(image)


def create_augmented_pair_blur(clean_image, mode = 'noisy'):
    """Create noisy versions based on blur"""

    image = np.array(clean_image)

    if mode == 'noisy':
        # Reduced the motion blur values, since it was too much and made the characters unrecognizable
        augmenter = iaa.Sequential(
            # Camera and motion effects
            iaa.OneOf([
                # Motion blur for vehicle movement
                iaa.MotionBlur(k=(5,10), angle=(-30, 30)),
                # Camera shake and focus issues
                iaa.GaussianBlur(sigma=(0.5, 3.0)),
                # Defocus blur
                iaa.AverageBlur(k=(2, 5))
            ]))
    else:
        # For the clean variations, use less strong blur
        augmenter = iaa.Sequential(
            # Camera and motion effects
            iaa.OneOf([
                # Motion blur for vehicle movement
                iaa.MotionBlur(k=(3,5), angle=(-15,15)),
                # Camera shake and focus issues
                iaa.GaussianBlur(sigma=(0.5, 1.5)),
                # Defocus blur
                iaa.AverageBlur(k=(2, 3))
            ]))


    image = augmenter(image=image)

    return Image.fromarray(image)


def create_augmented_pair_elastic_deform(clean_image, alpha_choice=(50,60,70,80,90,100)):
    """Simulate wrinkled/bent plates"""
    image = np.array(clean_image)

    image = cv2.bitwise_not(image)

    alpha = random.choice(alpha_choice)
    # 10:1 ratio according to https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html#imgaug.augmenters.geometric.ElasticTransformation
    augmenter = iaa.ElasticTransformation(alpha=alpha, sigma=int(alpha/10))

    image = augmenter(image=image)


    image = cv2.bitwise_not(image)
    

    return Image.fromarray(image)

'''

def clean_pipeline(clean_image):
    """Pipeline to create variations of the clean images"""

    augmented_image = clean_image

    augmentation_params = {'create_augmented_pair_imagequality' : {'compression_strength': (30,50)}, 
                                  'create_augmented_pair_rotation' : {'rotation_strength': (-3,3)},
                                  'create_augmented_pair_blur' : {'mode': 'clean'},
                                  'create_augmented_pair_edge_roughness' : {'strength_roughness_choice': (2,3,4)}}
    
    augmentation_possibilities = list(augmentation_params.keys())

    # Always apply atleast one 
    num_initial_augs = random.randint(1, 4)
    selected_initial = random.sample(augmentation_possibilities, num_initial_augs)

    # If edge roughness is selected, select it as the first one, as it works on a contour algorithm
    # (thus, if something else is first, might create some new dots and then it messes up the contour algorithm)
    if 'created_augmented_pair_edge_roughness' in selected_initial:
        selected_initial.remove('create_augmented_pair_edge_roughness')
        selected_initial.insert(0, 'create_augmented_pair_edge_roughness') 


    for aug_func_name in selected_initial:
        # Get the function from globals and apply it
        aug_func = globals()[aug_func_name]
        params = augmentation_params[aug_func_name]
        augmented_image = aug_func(augmented_image, **params)

    return augmented_image
'''


def get_all_clean_combinations(augmentation_possibilities):
    """Generate all possible clean augmentation combinations (1-4 augmentations)"""
    all_combinations = []
    for r in range(1, 5):  # 1 to 4 augmentations
        all_combinations.extend(list(itertools.combinations(augmentation_possibilities, r)))
    return all_combinations

def clean_pipeline(clean_image, char, path, n_examples = 10):
    """
    Creates all possible clean variation combinations and generates multiple examples
    for each combination, with preprocessing validation and character-specific subdirectories.
    """
    augmentation_params = {
        'create_augmented_pair_imagequality': {'compression_strength': (30,50)}, 
        'create_augmented_pair_rotation': {'rotation_strength': (-3,3)},
        'create_augmented_pair_blur': {'mode': 'clean'},
        'create_augmented_pair_edge_roughness': {'strength_roughness_choice': (2,3,4)}
    }
    
    augmentation_possibilities = list(augmentation_params.keys())
    
    # Get all possible combinations (always at least one augmentation)
    all_combinations = get_all_clean_combinations(augmentation_possibilities)

    # Define all possible characters
    all_characters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    # Create base directory if it doesn't exist
    base_path = Path(path)
    base_path.mkdir(exist_ok=True)

    # Process each combination
    for combo_idx, combination in enumerate(all_combinations):
        # Create directory for this combination
        combo_dir = base_path / f"clean_variation_{combo_idx}"
        combo_dir.mkdir(exist_ok=True)

        # Create subdirectories for all characters in this variation folder
        for character in all_characters:
            char_dir = combo_dir / character
            char_dir.mkdir(exist_ok=True)

        # Counter for successful generations
        successful_generations = 0
        example_idx = 0

        # Continue until we have 10 successful generations
        while successful_generations < n_examples:
            augmented_image = clean_image
            
            # Convert combination to list for potential reordering
            augs_to_apply = list(combination)
            
            # If edge roughness is in the combination, move it to the front
            if 'create_augmented_pair_edge_roughness' in augs_to_apply:
                augs_to_apply.remove('create_augmented_pair_edge_roughness')
                augs_to_apply.insert(0, 'create_augmented_pair_edge_roughness')

            # Apply the augmentations in order
            for aug_func_name in augs_to_apply:
                aug_func = globals()[aug_func_name]
                params = augmentation_params[aug_func_name]
                augmented_image = aug_func(augmented_image, **params)

            # Preprocess the augmented image
            processed_image = preprocessing(augmented_image)

            # Only save if preprocessing was successful
            if processed_image is not None:
                # Save the processed image in the character-specific subdirectory
                output_filename = f"{char}_{successful_generations}.png"
                output_path = combo_dir / char / output_filename
                processed_image.save(str(output_path))
                successful_generations += 1
            
            example_idx += 1

    return None

def get_all_possible_combinations(initial_augmentations, combination_options, single_options):
    """Generate all possible augmentation combinations"""
    # Get all possible initial combinations (including empty set)
    initial_combinations = []
    for r in range(5):  # 0 to 4 augmentations
        initial_combinations.extend(list(itertools.combinations(initial_augmentations, r)))
    
    # Combine with second stage options
    all_combinations = []
    
    # Initial augs + combination options
    for init in initial_combinations:
        for combo in combination_options:
            all_combinations.append((list(init), list(combo)))
            
    # Initial augs + single options
    for init in initial_combinations:
        for single in single_options:
            all_combinations.append((list(init), [single]))
    
    return all_combinations

def augmentation_pipeline(clean_image, char, path, n_examples = 10):
    """
    Creates all possible augmentation combinations and generates multiple examples
    for each combination, with preprocessing validation.
    
    Args:
        clean_image: Input image to be augmented
        char: Character being processed
        path: Base path where to create augmentation folders
    """
    # Define parameter ranges for each augmentation
    augmentation_params = {
        'create_augmented_pair_imagequality': {'compression_strength': (80,100)},
        'create_augmented_pair_rotation': {'rotation_strength': (-8,8)},
        'create_augmented_pair_blur': {'mode': 'noisy'},
        'create_augmented_pair_smushing': {'factor_choice': (0.4,0.5,0.6)},
        'create_augmented_pair_hole_filling': {},
        'create_augmented_pair_perspective_skew': {'scale': (0.05, 0.2)},
        'create_augmented_pair_thickness': {'area': (8,10)},
        'create_augmented_pair_elastic_deform': {'alpha_choice': (50,60,70,80,90,100)},
        'create_augmented_pair_occlusion': {
            'n_segments_choice': (7,8,9,10,11,12),
            'segment_length_choice': (7,8,9,10,11,12),
            'length_max_choice': tuple(np.arange(150, 200, 1))
        },
        'create_augmented_pair_edge_roughness': {'strength_roughness_choice': (6,7,8,9,10,11)}
    }

    # Define augmentation sets
    initial_augmentations = [
        'create_augmented_pair_imagequality',
        'create_augmented_pair_rotation',
        'create_augmented_pair_blur',
        'create_augmented_pair_edge_roughness'
    ]
    
    combination_options = [
        ['create_augmented_pair_smushing', 'create_augmented_pair_hole_filling'],
        ['create_augmented_pair_perspective_skew', 'create_augmented_pair_thickness'],
        ['create_augmented_pair_thickness', 'create_augmented_pair_occlusion'],
        ['create_augmented_pair_elastic_deform', 'create_augmented_pair_occlusion']
    ]
    
    single_options = [
        'create_augmented_pair_smushing',
        'create_augmented_pair_hole_filling',
        'create_augmented_pair_perspective_skew',
        'create_augmented_pair_thickness',
        'create_augmented_pair_elastic_deform',
        'create_augmented_pair_occlusion',
    ]

    # Get all possible combinations
    all_combinations = get_all_possible_combinations(
        initial_augmentations, 
        combination_options, 
        single_options
    )

    # Define all possible characters
    all_characters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    # Create base directory if it doesn't exist
    base_path = Path(path)
    base_path.mkdir(exist_ok=True)

    # Process each combination
    for combo_idx, (initial_augs, second_stage) in enumerate(all_combinations):
        # Create directory for this combination
        combo_dir = base_path / f"augmentation_m_{combo_idx}"
        combo_dir.mkdir(exist_ok=True)

        # Create subdirectories for all characters in this augmentation folder
        for character in all_characters:
            char_dir = combo_dir / character
            char_dir.mkdir(exist_ok=True)

        # Counter for successful generations
        successful_generations = 0
        example_idx = 0

        # Continue until we have 10 successful generations
        while successful_generations < n_examples:
            augmented_image = clean_image

            # Apply initial augmentations (if any)
            for aug_name in initial_augs:
                # Handle edge_roughness first if present
                if 'create_augmented_pair_edge_roughness' in initial_augs:
                    initial_augs.remove('create_augmented_pair_edge_roughness')
                    initial_augs.insert(0, 'create_augmented_pair_edge_roughness')
                
                aug_func = globals()[aug_name]
                params = augmentation_params[aug_name]
                augmented_image = aug_func(augmented_image, **params)

            # Apply second stage augmentations
            for aug_name in second_stage:
                aug_func = globals()[aug_name]
                params = augmentation_params[aug_name]
                augmented_image = aug_func(augmented_image, **params)

            # Check if elastic deform needs to be added
            has_elastic = any('elastic_deform' in aug for aug in initial_augs + second_stage)
            if not has_elastic:
                aug_name = 'create_augmented_pair_elastic_deform'
                aug_func = globals()[aug_name]
                params = augmentation_params[aug_name]
                augmented_image = aug_func(augmented_image, **params)

            # Preprocess the augmented image
            processed_image = preprocessing(augmented_image)

            # Only save if preprocessing was successful
            if processed_image is not None:
                # Save the processed image in the character-specific subdirectory
                output_filename = f"{char}_{successful_generations}.png"
                output_path = combo_dir / char / output_filename  # Note the added char subdirectory
                processed_image.save(str(output_path))
                successful_generations += 1
            
            example_idx += 1

    return None  # No need to return anything as images are saved directly


'''
def augmentation_pipeline(clean_image,char, i):
    """
    Applies a series of augmentations to an image according to specific rules:
    1. First applies 0-3 augmentations from the initial set randomly
    2. Then applies either a single augmentation or one of the predefined combinations
    
    Args:
        clean_image: Input image to be augmented
        
    Returns:
        Augmented image after pipeline processing
    """

    # Define parameter ranges for each augmentation
    augmentation_params = {
        'create_augmented_pair_imagequality': {'compression_strength': (80,100)},
        'create_augmented_pair_rotation': {'rotation_strength': (-8,8)},
        'create_augmented_pair_blur': {'mode': 'noisy'},
        'create_augmented_pair_smushing': {'factor_choice': (0.4,0.5,0.6)},
        'create_augmented_pair_hole_filling': {},
        'create_augmented_pair_perspective_skew': {'scale': (0.05, 0.2)},
        'create_augmented_pair_thickness': {'area': (8,10)},
        'create_augmented_pair_elastic_deform': {'alpha_choice': (50,60,70,80,90,100)},
        'create_augmented_pair_occlusion': {'n_segments_choice': (7,8,9,10,11,12), 'segment_length_choice': (7,8,9,10,11,12), 'length_max_choice': tuple(np.arange(150, 200, 1))},
        'create_augmented_pair_edge_roughness': {'strength_roughness_choice': (6,7,8,9,10,11)}
    }


    # Initial augmentation options
    initial_augmentations = [
        'create_augmented_pair_imagequality',
        'create_augmented_pair_rotation',
        'create_augmented_pair_blur',
        'create_augmented_pair_edge_roughness' # Need to be here so we can use it as first
    ]
    
    # Predefined combinations for second stage
    combination_options = [
        ['create_augmented_pair_smushing', 'create_augmented_pair_hole_filling'],
        ['create_augmented_pair_perspective_skew', 'create_augmented_pair_thickness'],
        ['create_augmented_pair_thickness', 'create_augmented_pair_occlusion'],
        ['create_augmented_pair_elastic_deform', 'create_augmented_pair_occlusion']

    ]
    
    # Additional single augmentation options for second stage
    single_options = [
        'create_augmented_pair_smushing',
        'create_augmented_pair_hole_filling',
        'create_augmented_pair_perspective_skew',
        'create_augmented_pair_thickness',
        'create_augmented_pair_elastic_deform',
        'create_augmented_pair_occlusion',
    ]
    
    # Initialize the image variable
    augmented_image = clean_image

    applications = str.upper(char) + "_" + str(i) + ".png : "
  #  print("STARTED")

    in_starter = False
    in_second_stage = False
    in_solo = False
    # Stage 1: Apply 0-3 initial augmentations randomly
    num_initial_augs = random.randint(0, 3)
    if num_initial_augs > 0:
        selected_initial = random.sample(initial_augmentations, num_initial_augs)
        # Select as first if it exists due to it using contour algorithm that messes up after other augmentations
        if 'create_augmented_pair_edge_roughness' in selected_initial:
            selected_initial.remove('create_augmented_pair_edge_roughness')
            selected_initial.insert(0, 'create_augmented_pair_edge_roughness')
    #    print(f"Selected initial augmentations: {selected_initial}")
        for aug_func_name in selected_initial:
            # Get the function from globals and apply it
            aug_func = globals()[aug_func_name]
            params = augmentation_params[aug_func_name]
            augmented_image = aug_func(augmented_image, **params)



            applications += aug_func_name + " "
            if aug_func_name == 'create_augmented_pair_elastic_deform':
                in_starter = True
    
    # Stage 2: Decide whether to apply a single augmentation or a combination
    apply_second_stage = True

    if apply_second_stage:
        # Decide between single augmentation or combination (50-50 chance)
        if random.random() < 0.5:
            # Apply single augmentation
            selected_aug = random.choice(single_options)
            aug_func = globals()[selected_aug]
            params = augmentation_params[selected_aug]
            augmented_image = aug_func(augmented_image, **params)
     #       print(f"Applied single augmentation: {selected_aug}")
            applications += selected_aug + " "
            if aug_func == 'create_augmented_pair_elastic_deform':
                in_solo = True
        else:
            # Apply combination
            selected_combo = random.choice(combination_options)
            for aug_func_name in selected_combo:
                aug_func = globals()[aug_func_name]
                params = augmentation_params[aug_func_name]
                augmented_image = aug_func(augmented_image, **params)
                applications += aug_func_name + " "
      #      print(f"Applied combination: {selected_combo}")
            if 'create_augmented_pair_elastic_deform' in selected_combo:
                in_second_stage = True
    
    # Finally, apply elastic deformation with high probability if not already applied
    if not in_starter and not in_second_stage and not in_solo:
       # Always apply elastic deformation at the end ; it helps making the character much more seem like it was extracted 
       # from actual noise
       aug_func_name = 'create_augmented_pair_elastic_deform'
       aug_func = globals()[aug_func_name]
       params = augmentation_params[aug_func_name]
       augmented_image = aug_func(augmented_image, **params)
       applications += aug_func_name + " "



        
  #  print("ENDED")



    return augmented_image, applications
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

    # New logic : 
    # 1. We know that the character is white and the background is black, i.e. pixel values are 0 for the background and >0 for the character
    # 2. To find a rectangle around the character, we just look at all pixels that are != 0 and find the bounding rectangle around them
    # That means : Find the white pixels that is most left in regard to all white pixels, most right, most top and most bottom
    # Use these to gather the 4 corner points of the rectangle and then crop the image to this rectangle
    # Then do the normal resizing steps

    # Find all white pixels

    white_pixels = np.where(image > 0)




    # Get the bounds
    try:
        min_y = np.min(white_pixels[0])  # top
        max_y = np.max(white_pixels[0])  # bottom
        min_x = np.min(white_pixels[1])  # left
        max_x = np.max(white_pixels[1])  # right
    except ValueError as e:
        print('Error finding bounding box ; skipping this image :', e)
        return None


    reg_of_interest = image[min_y:max_y, min_x:max_x] 

    # Get width and height of region of interest
    h, w = reg_of_interest.shape

    # Calculate scaling factor to fit in 20x20 box while maintaining aspect ratio
    try:
        scale = min(20.0/w, 20.0/h)
    except ZeroDivisionError as e:
        print('Error scaling the image ; skipping this image :', e)
        return None
    
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




def generate_dataset(output_dir, font_path):
    """Generate dataset of clean and noisy character images"""
    # Define character set (German license plates)
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    clean_dir = output_dir + '/clean'
    noisy_dir = output_dir + '/noisy'

    # First, clear text file if it exists
    if os.path.exists(noisy_dir + '/applications.txt'):
        os.remove(noisy_dir + '/applications.txt')

    # Create textfile to save applications
    with open(noisy_dir + '/applications.txt', 'w') as f:
        f.write('Applications:\n')
    
    # Generate multiple versions of each character
    for char in chars:

        # Generate clean image (just one since it's always the same ; TODO : possibly in different sizes could help, since license plate may be from diff perspectives?)
        clean_image = generate_single_character(char, font_path)
        # Preprocess image
     #   clean_image_processed = preprocessing(clean_image)
     #   clean_image_processed.save(clean_dir + f'/{char}/' + f'{char}_original.png')
        clean_pipeline(clean_image, char, clean_dir, n_examples=10)

     #   for i in range(100):  # Generate 100 clean versions
            # Generate clean variations
        #    clean_image_processed = None
      #      while clean_image_processed == None:

         #       clean_variation = clean_pipeline(clean_image)
        #        clean_image_processed = preprocessing(clean_variation)


            # Save images
        #    clean_image_processed.save(clean_dir + f'/{char}/' + f'{char}_{i}.png')
            
     #   for i in range(100):
            # Generate corresponding noisy image
        #    noisy_image_processed = None
         #   while noisy_image_processed == None:


        augmentation_pipeline(clean_image, char, noisy_dir, n_examples=10)
           #     noisy_image_processed = preprocessing(noisy_image)
                # Save applications
             #   with open(noisy_dir + '/applications.txt', 'a') as f:
             #       f.write(application + '\n')



            # Save images
         #   noisy_image_processed.save(noisy_dir + f'/{char}/' + f'{char}_{i}.png')






generate_dataset('VCS_Project/data/synthetic/german_font','VCS_Project/fonts/FE-FONT.TTF')




