#also create characters where some parts of the character are missing , like top part of a N or something like that for training (more robustness)


# !!! imgaug must be downloaded from this fork : https://github.com/marcown/imgaug

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import imgaug.augmenters as iaa
import os 
import cv2
import string
import torch
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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
            iaa.PerspectiveTransform(scale=(0.05, 0.15)),
            iaa.Affine(scale=(0.8, 1.2))
        ])),

        # Image quality and noise
        iaa.Sometimes(0.7, iaa.OneOf([
            iaa.AdditiveGaussianNoise(scale=(0, 0.15 * 255)),
            iaa.JpegCompression(compression=(70, 90)),
        ])),

        # Character degradation and occlusion
        iaa.Sometimes(0.6, iaa.OneOf([
            iaa.Dropout(p=(0.01, 0.1)),
            iaa.CoarseDropout(p=(0.02, 0.15), size_percent=(0.02, 0.05))
        ])),
    ])

    # Apply augmentation
    noisy_image = augmenter(image=image)
    
    return Image.fromarray(noisy_image)


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

    areas = [cv2.contourArea(contour) for contour in contours]
    max_area = max(areas)

    correct_img = None 
    for contour in sorted(contours, key=lambda x: cv2.boundingRect(x)[0]):
        x, y, w, h = cv2.boundingRect(contour)
    
        area = cv2.contourArea(contour)
        if area == max_area:
            reg_of_interest = image[y:y+h, x:x+w] 

            scale = min(20.0/w, 20.0/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            try:
                char_resized = cv2.resize(reg_of_interest, (new_w, new_h))
            except cv2.error as e:
                print('Error resizing the image ; skipping this image :', e)
                return None 
            
            mnist_size = np.zeros((28, 28), dtype=np.uint8)
            
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            
            mnist_size[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_resized
            correct_img = mnist_size

    return Image.fromarray(correct_img)

# Function to compute recognizability metrics
def compute_recognizability_metrics(image):
    """Calculate recognizability metrics for the given image."""
    edges = cv2.Canny(image, 100, 200)

    # Edge density
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    # Character stroke width consistency
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    stroke_widths = [cv2.boundingRect(contour)[2] for contour in contours if cv2.contourArea(contour) > 5]
    if len(stroke_widths) > 1:
        stroke_consistency = np.std(stroke_widths) / np.mean(stroke_widths)
    else:
        stroke_consistency = 1.0  # Default high inconsistency for edge cases

    # Bounding box eccentricity
    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        eccentricity = w / h if h != 0 else 1.0
    else:
        eccentricity = 1.0  # Default for edge cases

    # Pixel intensity histogram features
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram_variance = np.var(histogram)

    # Weighted sum of metrics
    recognizability_score = (
        0.5 * edge_density -
        0.3 * stroke_consistency +
        0.1 * eccentricity +
        0.1 * histogram_variance
    )

    return recognizability_score



# Define LeNet model for encoding
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Feature extractor
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x



# Define Autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self, pretrained_cnn):
        super(Autoencoder, self).__init__()
        # Use the encoder from the pretrained CNN (LeNet features)
        self.encoder = pretrained_cnn.features

        # Define a decoder that matches the encoder output
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 6, kernel_size=5, stride=2),  # Match 16 channels
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(6, 1, kernel_size=5, stride=2, output_padding=1),  # Back to 1 channel
            torch.nn.Sigmoid()  # Output normalized to [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction



def generate_dataset(output_dir, font_path):
    """Generate dataset of clean and noisy character images with improvements"""

    # Define character set (German license plates)
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    clean_dir = output_dir + '/clean'
    noisy_dir = output_dir + '/noisy'

    # Create main directories if they don't exist
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    # Load the pretrained CNN (LeNet)
    pretrained_cnn = LeNet()
    pretrained_cnn.load_state_dict(torch.load("LeNet/lenet_mnist_weights.pth", map_location=torch.device('cpu')))
    pretrained_cnn.eval()
    print("Pretrained CNN (LeNet) loaded.")

    # Initialize Autoencoder
    autoencoder = Autoencoder(pretrained_cnn)
    autoencoder.eval()
    print("Autoencoder initialized.")

    # Transformation pipeline
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Generate multiple versions of each character
    for char in chars:
        print(f"Processing character: {char}")

        # Ensure subdirectories for the character exist
        char_clean_dir = os.path.join(clean_dir, char)
        char_noisy_dir = os.path.join(noisy_dir, char)
        os.makedirs(char_clean_dir, exist_ok=True)
        os.makedirs(char_noisy_dir, exist_ok=True)

        # Generate and save clean images
        clean_image = generate_single_character(char, font_path)
        clean_image_processed = preprocessing(clean_image)
        clean_image_processed.save(os.path.join(char_clean_dir, f'{char}_original.png'))
        

        generated_clean_count = 0
        while generated_clean_count < 100:  # Generate 100 clean variations
            clean_variation = create_clean_variations(clean_image)
            clean_variation_processed = preprocessing(clean_variation)

            if clean_variation_processed is None:
                
                continue

            clean_variation_processed.save(os.path.join(char_clean_dir, f'{char}_{generated_clean_count}.png'))
            generated_clean_count += 1
            

        # Generate noisy images
        generated_noisy_count = 0
        while generated_noisy_count < 100:  # Ensure 100 recognizable noisy images
            noisy_scores = []
            processed_noisy_images = []

            # Generate extra noisy images and calculate metrics
            while len(noisy_scores) < 20:  # Generate a batch of 20 images
                noisy_image = create_augmented_pair(clean_image)
                noisy_image_processed = preprocessing(noisy_image)

                if noisy_image_processed is None:
                    print(f"Skipping unprocessable noisy image for {char}.")
                    continue

                noisy_image_np = np.array(noisy_image_processed)
                recognizability_score = compute_recognizability_metrics(noisy_image_np)
                noisy_scores.append(recognizability_score)
                processed_noisy_images.append(noisy_image_processed)

            # Determine the best GMM using BIC
            best_gmm = None
            lowest_bic = np.inf
            for n in range(2, 5):  # Test 2 to 4 clusters
                gmm = GaussianMixture(n_components=n, random_state=42)
                gmm.fit(np.array(noisy_scores).reshape(-1, 1))
                bic = gmm.bic(np.array(noisy_scores).reshape(-1, 1))
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
            gmm = best_gmm  # Use the best GMM

            # Get cluster probabilities and determine recognizability cluster
            cluster_probs = gmm.predict_proba(np.array(noisy_scores).reshape(-1, 1))
            cluster_means = gmm.means_.flatten()
            recognizability_cluster = np.argmax(cluster_means)

            # Filter images based on GMM probabilities
            for i, img in enumerate(processed_noisy_images):
                if generated_noisy_count >= 100:
                    break
                if cluster_probs[i][recognizability_cluster] > 0.9:  # Probability threshold
                    img.save(noisy_dir + f'/{char}/' + f'{char}_{generated_noisy_count}.png')
                    generated_noisy_count += 1
                    



    print("Dataset generation complete!")





generate_dataset('data/synthetic/german_font/LeNet', 'fonts/FE-FONT.TTF')

