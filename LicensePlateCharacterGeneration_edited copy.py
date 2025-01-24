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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

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

# LeNet feature extraction for image filtering
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        return features_flat

class RecognizabilityFilter:
    def __init__(self):
        self.feature_extractor = LeNet()
        state_dict = torch.load("LeNet/lenet_mnist_weights.pth", map_location=torch.device('cpu'))
        self.feature_extractor.load_state_dict(state_dict)
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def extract_features(self, images):
        features = []
        with torch.no_grad():
            for img in images:
                img_tensor = self.transform(img).unsqueeze(0)
                feature = self.feature_extractor(img_tensor)
                features.append(feature.squeeze().numpy())
        return np.array(features)

    def cluster_and_filter(self, clean_images, noisy_images, n_clusters=5):

        threshold = 0.1
        
        clean_features = self.extract_features(clean_images)
        noisy_features = self.extract_features(noisy_images)
        
        scaler = StandardScaler()
        clean_features_scaled = scaler.fit_transform(clean_features)
        noisy_features_scaled = scaler.transform(noisy_features)
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        cluster_labels = kmeans.fit_predict(noisy_features_scaled)
        
        filtered_clean = []
        filtered_noisy = []
        
        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_center = kmeans.cluster_centers_[cluster]
            distances = np.linalg.norm(noisy_features_scaled[cluster_indices] - cluster_center, axis=1)
            best_indices = cluster_indices[np.argsort(distances)[:5]]  # Select top 5 most representative
            
            filtered_clean.extend([clean_images[i] for i in best_indices])
            filtered_noisy.extend([noisy_images[i] for i in best_indices if self.compute_edge_quality([noisy_images[i]])[0] > threshold])
        
        return filtered_clean, filtered_noisy

    def compute_edge_quality(self, images):
        edge_qualities = []
        for img in images:
            img_array = np.array(img)
            edges = cv2.Canny(img_array, 100, 200)
            edge_density = np.sum(edges) / (img_array.shape[0] * img_array.shape[1])
            edge_qualities.append(edge_density)
        return edge_qualities

def generate_dataset(output_dir, font_path, num_per_character=100):
    create_directories(output_dir)
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    recognizability_filter = RecognizabilityFilter()

    for char in chars:
        clean_dir = os.path.join(output_dir, 'clean', char)
        noisy_dir = os.path.join(output_dir, 'noisy', char)
        
        clean_image = generate_single_character(char, font_path)
        clean_variations = [create_clean_variations(clean_image) for _ in range(num_per_character)]
        noisy_variations = [create_augmented_pair(clean_image) for _ in range(num_per_character)]
        
        clean_processed = [preprocessing(img) for img in clean_variations]
        noisy_processed = [preprocessing(img) for img in noisy_variations]
        
        clean_processed = [img for img in clean_processed if img is not None]
        noisy_processed = [img for img in noisy_processed if img is not None]
        
        filtered_clean, filtered_noisy = recognizability_filter.cluster_and_filter(clean_processed, noisy_processed)
        
        # Save the top 100 clean and noisy images for each character
        for i, (clean_img, noisy_img) in enumerate(zip(filtered_clean[:100], filtered_noisy[:100])):
            clean_img.save(os.path.join(clean_dir, f'{char}_{i}.png'))
            noisy_img.save(os.path.join(noisy_dir, f'{char}_{i}.png'))
    
    print("Enhanced dataset generation complete!")


# Use existing functions from the original script
generate_dataset('data/synthetic/german_font/lenet_dataset', 'fonts/FE-FONT.TTF')