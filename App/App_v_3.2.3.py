import dash
from dash import dcc, html, Input, Output, State
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import ollama
import os
from pathlib import Path

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variables to store images and history
uploaded_image = None
processed_image = None
history = []
redo_stack = []

# --------------------------------------------------------------------------------------------------------------------
# Helper functions 
# --------------------------------------------------------------------------------------------------------------------

# Image conversion function
def image_to_base64(image, format="jpg", pure_base64=False):
    """Converts an image to Base64 format. If `pure_base64=True`, removes metadata."""
    _, buffer = cv2.imencode(f'.{format}', image)
    encoded_image = base64.b64encode(buffer).decode()
    
    if pure_base64:
        return encoded_image  # Return raw Base64 for AI
    return f'data:image/{format};base64,{encoded_image}'  # Return with metadata for web display

# Valitade image function
def validate_image(image):
    """Validate the input image to ensure it's valid and not empty."""
    if image is None or not hasattr(image, 'size') or image.size == 0:
        raise ValueError("Invalid image: The input is empty or could not be loaded.")

# Image processing functions
def adjust_exposure(image, factor=1.0):
    """ Adjusts image exposure by scaling pixel values """
    validate_image(image)  # Validate input
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

# Adjusts image contrast
def adjust_contrast(image, factor=1.0):
    """ Adjusts image contrast using PIL """
    validate_image(image)  # Validate input
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_image)
    return cv2.cvtColor(np.array(enhancer.enhance(factor)), cv2.COLOR_RGB2BGR)

# Adjusts image brightness
def adjust_brightness(image, factor=0):
    """ Adjusts image brightness by adding a constant """
    validate_image(image)  # Validate input
    return cv2.convertScaleAbs(image, alpha=1, beta=factor)

# Thresholding function
def adaptive_thresholding(image, block_size=11, c_value=2):
    """ Applies adaptive thresholding to enhance readability """
    validate_image(image)  # Validate input
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)

# Edge detection function
def improved_edge_detection(image, min_threshold, max_threshold):
    """
    Enhanced edge detection using adaptive thresholding and gradients
    """
    validate_image(image)  # Validate input
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize the image
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(normalized, 11, 75, 75)
    
    # Calculate gradients using Sobel
    gradient_x = cv2.Sobel(bilateral, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(bilateral, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = np.uint8(gradient_magnitude * 255 / gradient_magnitude.max())
    
    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # C constant
    )
    
    # Combine gradient information with adaptive threshold
    combined = cv2.bitwise_and(gradient_magnitude, adaptive_thresh)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)


selected_points = []  # Global variable to store selected points

# Select corners function
def select_corners(img):
    """Simulate selecting corners by clicking on an image in the Dash UI."""
    global selected_points

    if len(selected_points) == 4:
        points_array = np.array(selected_points, dtype=np.float32)
        selected_points = []  # Reset after selection
        return points_array
    return None  # Not enough points selected yet

# Paint image function
def paint_image(img):
    """Allows users to draw on an image."""
    validate_image(img)  # Validate input
    drawing = False
    last_point = None
    img_display = img.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, last_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(img_display, last_point, (x, y), (0, 0, 255), 2)
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Paint Image")
    cv2.setMouseCallback("Paint Image", mouse_callback)

    while True:
        cv2.imshow("Paint Image", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save
            return img_display
        elif key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()
    return img

# Character segmentation function
def character_segmentation(image, deviation=10):
    """Segments individual characters from a license plate."""
    validate_image(image)  # Ensure the image is valid
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    character_images = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > deviation:
            char_img = gray[y:y+h, x:x+w]
            character_images.append(char_img)

    return character_images

    return character_images

# Perspective correction function
def perspective_correction(img=None, image_path=None):
    """Applies perspective transformation to correct skewed license plates."""
    if img is None and image_path:
        img = cv2.imread(image_path)

    validate_image(img)  # Ensure the image is valid
    src = select_corners(img)
    if src is None:
        return img

    height, width = img.shape[:2]
    dst = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    corrected_img = cv2.warpPerspective(img, M, (width, height))

    return corrected_img

def on_change(_):
    """Updates adaptive thresholding interactively."""
    pass  # Placeholder for trackbar callback

def on_brush_change(_):
    """Handles brush size adjustments while painting."""
    pass  # Placeholder for trackbar callback

# Automatic license plate detection and cropping
def auto_crop_license_plate(image):
    validate_image(image)  # Validate input
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
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    return image

# --------------------------------------------------------------------------------------------------------------------
# Layout of the app 
# --------------------------------------------------------------------------------------------------------------------
app.layout = html.Div(
    style={"backgroundColor": "#1f1f1f", "color": "#f0f0f0", "fontFamily": "Arial"},
    children=[
        # Header
        html.Div(
            style={"padding": "20px", "textAlign": "center", "backgroundColor": "#333333"},
            children=[
                html.H1("Interactive Image Processing", style={"color": "#ffcc00"}),
                html.P("Upload and process images with real-time adjustments.")
            ]
        ),

        # Main content area
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "padding": "20px"},
            children=[
                # Sidebar for controls
                html.Div(
                    style={"width": "30%", "backgroundColor": "#2a2a2a", "padding": "15px", "borderRadius": "10px"},
                    children=[
                        html.H3("Controls", style={"textAlign": "center"}),

                        dcc.Upload(
                            id='upload-image',
                            children=html.A('Upload Image', 
                                              style={"display": "block", "padding": "10px", "backgroundColor": "#007bff", "color": "#fff", "border": "none", "borderRadius": "5px", "textDecoration": "none", "textAlign": "center"}),
                            multiple=False
                        ),

                        html.Button('Auto Crop License Plate', id='auto-crop-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#8a00dc", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "20px"}),

                        html.Button("Select Corners", id="select-corners-btn", n_clicks=0, 
                                    style={"width": "49%", "padding": "10px", "backgroundColor": "#28a745", "color": "#fff", 
                                        "border": "none", "borderRadius": "5px", "marginTop": "10px", "marginRight": "2%"}),

                        html.Button("Perspective Correction", id="perspective-correction-btn", n_clicks=0, 
                                    style={"width": "49%", "padding": "10px", "backgroundColor": "#17a2b8", "color": "#fff", 
                                        "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button("Paint Image", id="paint-image-btn", n_clicks=0, 
                                    style={"width": "49%", "padding": "10px", "backgroundColor": "#dc3545", "color": "#fff", 
                                        "border": "none", "borderRadius": "5px", "marginTop": "10px", "marginRight": "2%"}),

                        html.Button("Segment Characters", id="segment-characters-btn", n_clicks=0, 
                                    style={"width": "49%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", 
                                        "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        dcc.Store(id="stored-image-path"),  # Store image path for processing


                        html.Button('Exposure Options', id='toggle-exposure-options', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#ff8700", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "20px"}),

                        html.Div(id='exposure-options', style={"display": "none", "marginTop": "20px"}, children=[
                            html.Div("Exposure Adjustment", style={"marginBottom": "10px"}),
                            dcc.Slider(
                                id='exposure-slider',
                                min=0.5,
                                max=2.0,
                                step=0.1,
                                value=1.0,
                                marks={i/10: str(i/10) for i in range(5, 21, 5)},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Button('Apply Exposure', id='exposure-btn', n_clicks=0, 
                                      style={"width": "100%", "padding": "10px", "backgroundColor": "#ff8700", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        html.Button('Contrast Options', id='toggle-contrast-options', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#8a00dc", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Div(id='contrast-options', style={"display": "none", "marginTop": "20px"}, children=[
                            html.Div("Contrast Adjustment", style={"marginBottom": "10px"}),
                            dcc.Slider(
                                id='contrast-slider',
                                min=0.5,
                                max=2.0,
                                step=0.1,
                                value=1.0,
                                marks={i/10: str(i/10) for i in range(5, 21, 5)},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Button('Apply Contrast', id='contrast-btn', n_clicks=0, 
                                      style={"width": "100%", "padding": "10px", "backgroundColor": "#8a00dc", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        html.Button('Brightness Options', id='toggle-brightness-options', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#dc3545", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Div(id='brightness-options', style={"display": "none", "marginTop": "20px"}, children=[
                            html.Div("Brightness Adjustment", style={"marginBottom": "10px"}),
                            dcc.Slider(
                                id='brightness-slider',
                                min=-50,
                                max=50,
                                step=1,
                                value=0,
                                marks={i: str(i) for i in range(-50, 51, 25)},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Button('Apply Brightness', id='brightness-btn', n_clicks=0, 
                                      style={"width": "100%", "padding": "10px", "backgroundColor": "#dc3545", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        html.Button('Edge Detection Options', id='toggle-threshold-options', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Div(id='threshold-options', style={"display": "none", "marginTop": "20px"}, children=[
                            html.Div("Threshold Block Size", style={"marginBottom": "10px"}),
                            dcc.Slider(
                                id='threshold-block-slider',
                                min=3,
                                max=51,
                                step=2,
                                value=11,
                                marks={i: str(i) for i in range(3, 52, 6)},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Button('Apply Threshold', id='threshold-btn', n_clicks=0, 
                                      style={"width": "100%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        
                        html.Button('Blur Options', id='toggle-blur-options', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#17a2b8", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Div(id='blur-options', style={"display": "none", "marginTop": "20px"}, children=[
                            html.Div("Blur Kernel Size", style={"marginBottom": "10px"}),
                            dcc.Slider(
                                id='blur-slider',
                                min=1,
                                max=11,
                                step=2,
                                value=3,
                                marks={i: f'{i}px' for i in range(1, 12, 2)},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Button('Apply Blur', id='blur-btn', n_clicks=0, 
                                      style={"width": "100%", "padding": "10px", "backgroundColor": "#17a2b8", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

#                        html.Div("Edge Detection Thresholds", style={"marginTop": "20px"}),
#                        dcc.Slider(id='threshold1-slider', min=0, max=255, step=1, value=75, 
#                                 marks={i: str(i) for i in range(0, 256, 50)}, tooltip={"placement": "bottom"}),
#                        dcc.Slider(id='threshold2-slider', min=0, max=255, step=1, value=175, 
#                                 marks={i: str(i) for i in range(0, 256, 50)}, tooltip={"placement": "bottom"}),
#                        html.Button('Edge Detection', id='edge-btn', n_clicks=0, 
#                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Grayscale', id='grayscale-btn', n_clicks=0, 
                                  style={"width": "32%", "padding": "10px", "backgroundColor": "#6c757d", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "20px", "marginRight": "2%"}),
                        html.Button('Sharpen', id='sharpen-btn', n_clicks=0, 
                                  style={"width": "32%", "padding": "10px", "backgroundColor": "#007bff", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px", "marginRight": "2%"}),
                        html.Button('Invert Colors', id='invert-btn', n_clicks=0, 
                                  style={"width": "32%", "padding": "10px", "backgroundColor": "#dc3545", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),
                        
#                        html.Button('Edge Detection', id='edge-btn', n_clicks=0, 
#                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),
                       
                        html.Button('Undo', id='undo-btn', n_clicks=0, 
                                  style={"width": "49%", "padding": "10px", "backgroundColor": "#6c757d", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px", "marginRight": "2%"}),
                        html.Button('Redo', id='redo-btn', n_clicks=0, 
                                  style={"width": "49%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.A(
                            "Download Processed Image", id="download-link", download="processed_image.png", 
                            href="", target="_blank",
                            style={"display": "block", "textAlign": "center", "marginTop": "20px", "padding": "10px", "backgroundColor": "#007bff", "color": "#fff", "border": "none", "borderRadius": "5px", "textDecoration": "none"}
                        )
                    ]
                ),

                # Main display for the image
                html.Div(
                    style={"width": "65%", "backgroundColor": "#2a2a2a", "padding": "15px", "borderRadius": "10px", "textAlign": "center"},
                    children=[
                        html.H3("Image Preview", style={"textAlign": "center"}),
                        html.Div(id='image-display', style={"marginTop": "20px"})
                    ]
                )
            ]
        ),

        # Footer
        html.Div(
            style={"padding": "10px", "textAlign": "center", "backgroundColor": "#333333"},
            children=html.P("Developed by Alberto Calabrese, Marlon Helbing and Daniele Virzì, 2025", style={"margin": 0})
        )
    ]
)

# --------------------------------------------------------------------------------------------------------------------
# Callback for image processing
# --------------------------------------------------------------------------------------------------------------------

@app.callback(
    [
        Output('image-display', 'children'),
        Output('blur-options', 'style'),
        Output('exposure-options', 'style'),
        Output('contrast-options', 'style'),
        Output('brightness-options', 'style'),
        Output('threshold-options', 'style'),
        Output('download-link', 'href')
    ],
    [
        Input('upload-image', 'contents'),
        Input('auto-crop-btn', 'n_clicks'),
        Input('blur-btn', 'n_clicks'),
        Input('exposure-btn', 'n_clicks'),
        Input('contrast-btn', 'n_clicks'),
        Input('brightness-btn', 'n_clicks'),
        Input('threshold-btn', 'n_clicks'),
        Input('grayscale-btn', 'n_clicks'),
        Input('sharpen-btn', 'n_clicks'),
        Input('invert-btn', 'n_clicks'),
        Input('undo-btn', 'n_clicks'),
        Input('redo-btn', 'n_clicks'),
        Input('select-corners-btn', 'n_clicks'),
        Input('perspective-correction-btn', 'n_clicks'),
        Input('segment-characters-btn', 'n_clicks'),
        Input('toggle-blur-options', 'n_clicks'),
        Input('toggle-exposure-options', 'n_clicks'),
        Input('toggle-contrast-options', 'n_clicks'),
        Input('toggle-brightness-options', 'n_clicks'),
        Input('toggle-threshold-options', 'n_clicks'),
    ],
    [
        State('blur-slider', 'value'),
        State('blur-options', 'style'),
        State('exposure-slider', 'value'),
        State('exposure-options', 'style'),
        State('contrast-slider', 'value'),
        State('contrast-options', 'style'),
        State('brightness-slider', 'value'),
        State('brightness-options', 'style'),
        State('threshold-block-slider', 'value'),
        State('threshold-options', 'style'),
    ],
)

# --------------------------------------------------------------------------------------------------------------------
# Processing the image
# --------------------------------------------------------------------------------------------------------------------

def process_image(
    contents, auto_crop_clicks, blur_clicks, exposure_clicks, contrast_clicks, brightness_clicks,
    threshold_clicks, grayscale_clicks, sharpen_clicks, invert_clicks, undo_clicks, redo_clicks,
    select_corners_clicks, perspective_correction_clicks,
    segment_characters_clicks,
    toggle_blur_clicks, toggle_exposure_clicks, toggle_contrast_clicks, toggle_brightness_clicks,
    toggle_threshold_clicks,
    blur_kernel, blur_style, exposure_value, exposure_style, contrast_value, contrast_style,
    brightness_value, brightness_style, threshold_block, threshold_style
):
    global uploaded_image, processed_image, history, redo_stack

    ctx = dash.callback_context
    if not ctx.triggered:
        return (
            None, blur_style, exposure_style, contrast_style, brightness_style, threshold_style, "", dash.no_update
        )

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    suggestion_output = dash.no_update

    # Handle image upload
    if trigger_id == 'upload-image':
        if contents:
            _, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            np_image = np.frombuffer(decoded, np.uint8)
            uploaded_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            processed_image = uploaded_image.copy()
            history = [processed_image.copy()]
            redo_stack = []
            encoded_image = image_to_base64(uploaded_image, format="png")
            return (
                html.Img(src=encoded_image, style={"width": "90%", "borderRadius": "10px"}),
                blur_style, exposure_style, contrast_style, brightness_style, threshold_style,
                f"data:image/png;base64,{encoded_image.split(',')[1]}", dash.no_update
            )

    # Validate `processed_image` before applying transformations
    if processed_image is None:
        return (
            html.Div("Error: No image to process. Please upload an image first.", style={"color": "red"}),
            blur_style, exposure_style, contrast_style, brightness_style, threshold_style,
            "", dash.no_update
        )

    # Toggle option panels
    toggle_dict = {
        'toggle-blur-options': blur_style,
        'toggle-exposure-options': exposure_style,
        'toggle-contrast-options': contrast_style,
        'toggle-brightness-options': brightness_style,
        'toggle-threshold-options': threshold_style
    }

    if trigger_id in toggle_dict:
        current_style = toggle_dict[trigger_id]
        new_style = {"display": "block"} if current_style == {"display": "none"} else {"display": "none"}
        return (
            dash.no_update,
            new_style if trigger_id == 'toggle-blur-options' else blur_style,
            new_style if trigger_id == 'toggle-exposure-options' else exposure_style,
            new_style if trigger_id == 'toggle-contrast-options' else contrast_style,
            new_style if trigger_id == 'toggle-brightness-options' else brightness_style,
            new_style if trigger_id == 'toggle-threshold-options' else threshold_style,
            "", dash.no_update
        )

    # Apply transformations
    if processed_image is not None:
        if trigger_id == 'auto-crop-btn':
            processed_image = auto_crop_license_plate(processed_image)
        elif trigger_id == 'blur-btn':
            blur_kernel = max(1, blur_kernel // 2 * 2 + 1)  # Ensure odd kernel size
            processed_image = cv2.GaussianBlur(processed_image, (blur_kernel, blur_kernel), 0)
        elif trigger_id == 'exposure-btn':
            processed_image = adjust_exposure(processed_image, exposure_value)
        elif trigger_id == 'contrast-btn':
            processed_image = adjust_contrast(processed_image, contrast_value)
        elif trigger_id == 'brightness-btn':
            processed_image = adjust_brightness(processed_image, brightness_value)
        elif trigger_id == 'threshold-btn':
            processed_image = adaptive_thresholding(processed_image, threshold_block, 2)
        elif trigger_id == 'grayscale-btn':
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        elif trigger_id == 'sharpen-btn':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_image = cv2.filter2D(processed_image, -1, kernel)
        elif trigger_id == 'invert-btn':
            processed_image = cv2.bitwise_not(processed_image)
        elif trigger_id == 'select-corners-btn':
            processed_image = select_corners(processed_image)
        elif trigger_id == 'perspective-correction-btn':
            processed_image = perspective_correction(img=processed_image)
        elif trigger_id == 'segment-characters-btn':
            characters = character_segmentation(processed_image)


        # Save history for undo functionality
        if len(history) == 0 or not np.array_equal(history[-1], processed_image):
            history.append(processed_image.copy())
            redo_stack.clear()

        # Undo functionality
        if trigger_id == 'undo-btn' and len(history) > 1:
            redo_stack.append(history.pop())
            processed_image = history[-1].copy()

        # Redo functionality
        if trigger_id == 'redo-btn' and redo_stack:
            processed_image = redo_stack.pop()
            history.append(processed_image.copy())

    # Convert processed image to base64 for display
    encoded_image = image_to_base64(processed_image, format="png")
    download_href = f"data:image/png;base64,{encoded_image.split(',')[1]}"
    return (
        html.Img(src=encoded_image, style={"width": "90%", "borderRadius": "10px"}),
        blur_style, exposure_style, contrast_style, brightness_style, threshold_style,
        download_href, suggestion_output
    )


# --------------------------------------------------------------------------------------------------------------------
# Run the app
# --------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)