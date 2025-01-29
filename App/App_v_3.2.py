import dash
from dash import dcc, html, Input, Output, State
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageEnhance

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

# Image processing functions
def adjust_exposure(image, factor=1.0):
    """ Adjusts image exposure by scaling pixel values """
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_contrast(image, factor=1.0):
    """ Adjusts image contrast using PIL """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_image)
    return cv2.cvtColor(np.array(enhancer.enhance(factor)), cv2.COLOR_RGB2BGR)

def adjust_brightness(image, factor=0):
    """ Adjusts image brightness by adding a constant """
    return cv2.convertScaleAbs(image, alpha=1, beta=factor)

def improved_edge_detection(image):
    """
    Enhanced edge detection using adaptive thresholding and gradients
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    bilateral = cv2.bilateralFilter(normalized, 11, 75, 75)
    gradient_x = cv2.Sobel(bilateral, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(bilateral, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = np.uint8(gradient_magnitude * 255 / gradient_magnitude.max())
    
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # C constant
    )
    
    combined = cv2.bitwise_and(gradient_magnitude, adaptive_thresh)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

# Layout of the app
app.layout = html.Div(
    style={"backgroundColor": "#1f1f1f", "color": "#f0f0f0", "fontFamily": "Arial"},
    children=[
        html.Div(
            style={"padding": "20px", "textAlign": "center", "backgroundColor": "#333333"},
            children=[
                html.H1("Interactive Image Processing", style={"color": "#ffcc00"}),
                html.P("Upload and process images with real-time adjustments.")
            ]
        ),
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "padding": "20px"},
            children=[
                html.Div(
                    style={"width": "30%", "backgroundColor": "#2a2a2a", "padding": "15px", "borderRadius": "10px"},
                    children=[
                        html.H3("Controls", style={"textAlign": "center"}),

                        dcc.Upload(
                            id='upload-image',
                            children=html.Button('Upload Image', style={"width": "100%", "padding": "10px"}),
                            multiple=False
                        ),
                        html.Div("Exposure", style={"marginTop": "20px"}),
                        dcc.Slider(id='exposure-slider', min=0.5, max=2.0, step=0.1, value=1.0),
                        
                        html.Div("Contrast", style={"marginTop": "20px"}),
                        dcc.Slider(id='contrast-slider', min=0.5, max=2.0, step=0.1, value=1.0),

                        html.Div("Brightness", style={"marginTop": "20px"}),
                        dcc.Slider(id='brightness-slider', min=-50, max=50, step=1, value=0),

                        html.Button('Apply Edge Detection', id='edge-btn', n_clicks=0, 
                                    style={"width": "100%", "padding": "10px", "backgroundColor": "#ffc107", "marginTop": "10px"}),

                        html.Button('Undo', id='undo-btn', n_clicks=0, style={"width": "48%", "marginTop": "10px"}),
                        html.Button('Redo', id='redo-btn', n_clicks=0, style={"width": "48%", "marginTop": "10px"}),

                        html.A("Download Processed Image", id="download-link", download="processed_image.png", href="", target="_blank")
                    ]
                ),
                html.Div(
                    style={"width": "65%", "backgroundColor": "#2a2a2a", "padding": "15px", "borderRadius": "10px"},
                    children=[
                        html.H3("Image Preview", style={"textAlign": "center"}),
                        html.Div(id='image-display', style={"marginTop": "20px"})
                    ]
                )
            ]
        )
    ]
)

@app.callback(
    [Output('image-display', 'children'),
     Output('download-link', 'href')],
    [Input('upload-image', 'contents'),
     Input('exposure-slider', 'value'),
     Input('contrast-slider', 'value'),
     Input('brightness-slider', 'value'),
     Input('edge-btn', 'n_clicks'),
     Input('undo-btn', 'n_clicks'),
     Input('redo-btn', 'n_clicks')],
    [State('image-display', 'children')]
)
def process_image(contents, exposure_value, contrast_value, brightness_value, 
                  edge_clicks, undo_clicks, redo_clicks, image_display):
    global uploaded_image, processed_image, history, redo_stack

    ctx = dash.callback_context
    if not ctx.triggered:
        return None, ""

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
    
    if processed_image is not None:
        base_image = history[-1].copy()
        temp_image = adjust_exposure(base_image, exposure_value)
        temp_image = adjust_contrast(temp_image, contrast_value)
        temp_image = adjust_brightness(temp_image, brightness_value)
        
        if trigger_id == 'edge-btn':
            temp_image = improved_edge_detection(temp_image)

        processed_image = temp_image
        history.append(processed_image.copy())
        redo_stack.clear()

    if trigger_id == 'undo-btn' and len(history) > 1:
        redo_stack.append(history.pop())
        processed_image = history[-1].copy()

    if trigger_id == 'redo-btn' and redo_stack:
        processed_image = redo_stack.pop()
        history.append(processed_image.copy())

    encoded_image = image_to_base64(processed_image, format="png")
    download_href = f"data:image/png;base64,{encoded_image.split(',')[1]}"
    
    return html.Img(src=encoded_image, style={"width": "90%", "borderRadius": "10px"}), download_href

if __name__ == '__main__':
    app.run_server(debug=True)
