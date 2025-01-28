import dash
from dash import dcc, html, Input, Output, State
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variables to store images and history
uploaded_image = None
processed_image = None
history = []
redo_stack = []

# Helper function to convert image to base64
def image_to_base64(image, format="jpg"):
    _, buffer = cv2.imencode(f'.{format}', image)
    encoded_image = base64.b64encode(buffer).decode()
    return f'data:image/{format};base64,{encoded_image}'

# Automatic license plate detection and cropping
def auto_crop_license_plate(image):
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

    return image  # Return the original image if no license plate is detected

# Layout of the app
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
                            children=html.Button('Upload Image', 
                                                 style={"width": "100%", "padding": "10px", "backgroundColor": "#007bff", "color": "#fff", "border": "none", "borderRadius": "5px"}),
                            multiple=False
                        ),

                        html.Button('Auto Crop License Plate', id='auto-crop-btn', n_clicks=0, 
                                    style={"width": "100%", "padding": "10px", "backgroundColor": "#28a745", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

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

                        html.Div("Edge Detection Thresholds", style={"marginTop": "20px"}),
                        dcc.Slider(id='threshold1-slider', min=0, max=255, step=1, value=75, 
                                   marks={i: str(i) for i in range(0, 256, 50)}, tooltip={"placement": "bottom"}),
                        dcc.Slider(id='threshold2-slider', min=0, max=255, step=1, value=175, 
                                   marks={i: str(i) for i in range(0, 256, 50)}, tooltip={"placement": "bottom"}),
                        html.Button('Edge Detection', id='edge-btn', n_clicks=0, 
                                    style={"width": "100%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Grayscale', id='grayscale-btn', n_clicks=0, 
                                    style={"width": "100%", "padding": "10px", "backgroundColor": "#6c757d", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),
                        html.Button('Sharpen', id='sharpen-btn', n_clicks=0, 
                                    style={"width": "100%", "padding": "10px", "backgroundColor": "#007bff", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),
                        html.Button('Invert Colors', id='invert-btn', n_clicks=0, 
                                    style={"width": "100%", "padding": "10px", "backgroundColor": "#dc3545", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Undo', id='undo-btn', n_clicks=0, 
                                    style={"width": "48%", "padding": "10px", "backgroundColor": "#6c757d", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px", "marginRight": "2%"}),
                        html.Button('Redo', id='redo-btn', n_clicks=0, 
                                    style={"width": "48%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

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
            children=html.P("Developed Alberto Calabrese, Marlon Helbing and Daniele Virzì, 2025", style={"margin": 0})
        )
    ]
)

def improved_edge_detection(image, min_threshold, max_threshold):
    """
    Enhanced edge detection using adaptive thresholding and gradients
    """
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
    
    # Apply threshold based on slider values
    _, final = cv2.threshold(combined, min_threshold, max_threshold, cv2.THRESH_BINARY)
    
    # Clean up noise
    kernel = np.ones((2,2),np.uint8)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)
    
    return final

# Modify the process_image callback to use the new function
@app.callback(
    [
        Output('image-display', 'children'),
        Output('blur-options', 'style'),
        Output('download-link', 'href')
    ],
    [
        Input('upload-image', 'contents'),
        Input('auto-crop-btn', 'n_clicks'),
        Input('blur-btn', 'n_clicks'),
        Input('edge-btn', 'n_clicks'),
        Input('grayscale-btn', 'n_clicks'),
        Input('sharpen-btn', 'n_clicks'),
        Input('invert-btn', 'n_clicks'),
        Input('undo-btn', 'n_clicks'),
        Input('redo-btn', 'n_clicks'),
        Input('toggle-blur-options', 'n_clicks')
    ],
    [
        State('blur-slider', 'value'),
        State('threshold1-slider', 'value'),
        State('threshold2-slider', 'value'),
        State('blur-options', 'style')
    ]
)
def process_image(contents, auto_crop_clicks, blur_clicks, edge_clicks, grayscale_clicks, 
                 sharpen_clicks, invert_clicks, undo_clicks, redo_clicks, toggle_blur_clicks, 
                 blur_kernel, threshold1, threshold2, blur_style):
    global uploaded_image, processed_image, history, redo_stack

    ctx = dash.callback_context
    if not ctx.triggered:
        return None, blur_style, ""

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

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
                blur_style, 
                f"data:image/png;base64,{encoded_image.split(',')[1]}"
            )

    if trigger_id == 'toggle-blur-options':
        if blur_style == {"display": "none"}:
            return dash.no_update, {"display": "block"}, ""
        else:
            return dash.no_update, {"display": "none"}, ""

    if trigger_id == 'undo-btn':
        if len(history) > 1:
            redo_stack.append(history.pop())
            processed_image = history[-1].copy()
            encoded_image = image_to_base64(processed_image, format="png")
            return (
                html.Img(src=encoded_image, style={"width": "90%", "borderRadius": "10px"}), 
                blur_style, 
                f"data:image/png;base64,{encoded_image.split(',')[1]}"
            )

    if trigger_id == 'redo-btn':
        if redo_stack:
            processed_image = redo_stack.pop()
            history.append(processed_image.copy())
            encoded_image = image_to_base64(processed_image, format="png")
            return (
                html.Img(src=encoded_image, style={"width": "90%", "borderRadius": "10px"}), 
                blur_style, 
                f"data:image/png;base64,{encoded_image.split(',')[1]}"
            )

    if trigger_id == 'auto-crop-btn':
        if processed_image is not None:
            processed_image = auto_crop_license_plate(processed_image)
            history.append(processed_image.copy())
            redo_stack.clear()

    if trigger_id == 'blur-btn':
        if processed_image is not None:
            blur_kernel = max(1, blur_kernel // 2 * 2 + 1)
            processed_image = cv2.GaussianBlur(processed_image, (blur_kernel, blur_kernel), 0)
            history.append(processed_image.copy())
            redo_stack.clear()

    if trigger_id == 'edge-btn':
        if processed_image is not None:
            min_thresh = int((threshold1 / 255.0) * 100)  # Scale to 0-100 range
            max_thresh = 255  # Keep maximum brightness
            edges = improved_edge_detection(processed_image, min_thresh, max_thresh)
            processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            history.append(processed_image.copy())
            redo_stack.clear()

    if trigger_id == 'grayscale-btn':
        if processed_image is not None:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            history.append(processed_image.copy())
            redo_stack.clear()

    if trigger_id == 'sharpen-btn':
        if processed_image is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_image = cv2.filter2D(processed_image, -1, kernel)
            history.append(processed_image.copy())
            redo_stack.clear()

    if trigger_id == 'invert-btn':
        if processed_image is not None:
            processed_image = cv2.bitwise_not(processed_image)
            history.append(processed_image.copy())
            redo_stack.clear()

    # Convert the processed image to base64 for download
    encoded_image = image_to_base64(processed_image, format="png")
    download_href = f"data:image/png;base64,{encoded_image.split(',')[1]}"
    
    return (
        html.Img(src=image_to_base64(processed_image), style={"width": "90%", "borderRadius": "10px"}), 
        blur_style, 
        download_href
    )

'''
def download_image(n_clicks):
    global processed_image
    if processed_image is not None:
        # Convert the image to RGB before saving to ensure compatibility
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_image_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        return f'data:image/png;base64,{encoded_image}'
    return ""
'''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)