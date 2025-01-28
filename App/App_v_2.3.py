import dash
from dash import dcc, html, Input, Output, State
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(
    style={"backgroundColor": "#2c2c2c", "color": "white", "fontFamily": "Arial"},
    children=[
        html.H1("Interactive Image Processing App", style={"textAlign": "center", "color": "#ffcc00"}),

        # File upload component
        dcc.Upload(
            id='upload-image',
            children=html.Button('Upload Image', style={"fontSize": "16px", "padding": "10px 20px", "color": "#fff", "backgroundColor": "#007bff", "border": "none", "borderRadius": "5px"}),
            multiple=False
        ),

        # Display uploaded image
        html.Div(id='image-display', style={"textAlign": "center", "marginTop": "20px"}),

        # Controls for processing
        html.Div(
            style={"textAlign": "center", "marginTop": "20px"},
            children=[
                html.Div("Crop Ratio", style={"margin": "10px 0"}),
                dcc.Slider(id='crop-slider', min=0, max=50, step=1, value=10, 
                           marks={i: f'{i}%' for i in range(0, 51, 10)}, tooltip={"placement": "bottom"}),
                html.Button('Crop & Resize', id='crop-btn', n_clicks=0, style={"margin": "10px", "padding": "10px 20px", "color": "#fff", "backgroundColor": "#28a745", "border": "none", "borderRadius": "5px"}),

                html.Div("Blur Kernel Size", style={"margin": "10px 0"}),
                dcc.Slider(id='blur-slider', min=1, max=21, step=2, value=5, 
                           marks={i: str(i) for i in range(1, 22, 2)}, tooltip={"placement": "bottom"}),
                html.Button('Blur', id='blur-btn', n_clicks=0, style={"margin": "10px", "padding": "10px 20px", "color": "#fff", "backgroundColor": "#17a2b8", "border": "none", "borderRadius": "5px"}),

                html.Div("Edge Detection Thresholds", style={"margin": "10px 0"}),
                dcc.Slider(id='threshold1-slider', min=0, max=255, step=1, value=50, 
                           marks={i: str(i) for i in range(0, 256, 50)}, tooltip={"placement": "bottom"}),
                dcc.Slider(id='threshold2-slider', min=0, max=255, step=1, value=150, 
                           marks={i: str(i) for i in range(0, 256, 50)}, tooltip={"placement": "bottom"}),
                html.Button('Edge Detection', id='edge-btn', n_clicks=0, style={"margin": "10px", "padding": "10px 20px", "color": "#fff", "backgroundColor": "#ffc107", "border": "none", "borderRadius": "5px"}),

                html.Button('Grayscale', id='grayscale-btn', n_clicks=0, style={"margin": "10px", "padding": "10px 20px", "color": "#fff", "backgroundColor": "#6c757d", "border": "none", "borderRadius": "5px"}),
                html.Button('Sharpen', id='sharpen-btn', n_clicks=0, style={"margin": "10px", "padding": "10px 20px", "color": "#fff", "backgroundColor": "#007bff", "border": "none", "borderRadius": "5px"}),
                html.Button('Invert Colors', id='invert-btn', n_clicks=0, style={"margin": "10px", "padding": "10px 20px", "color": "#fff", "backgroundColor": "#dc3545", "border": "none", "borderRadius": "5px"}),
            ]
        ),

        # Download button
        html.A(
            "Download Processed Image", id="download-link", download="processed_image.jpg", 
            href="", target="_blank",
            style={"display": "block", "textAlign": "center", "marginTop": "20px", "padding": "10px 20px", "color": "#fff", "backgroundColor": "#007bff", "border": "none", "borderRadius": "5px", "textDecoration": "none"}
        )
    ]
)

# Global variables to store images
uploaded_image = None
processed_image = None

# Helper function to convert image to base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode()
    return f'data:image/jpg;base64,{encoded_image}'

# Consolidated callback for processing
@app.callback(
    Output('image-display', 'children'),
    [
        Input('upload-image', 'contents'),
        Input('crop-btn', 'n_clicks'),
        Input('blur-btn', 'n_clicks'),
        Input('edge-btn', 'n_clicks'),
        Input('grayscale-btn', 'n_clicks'),
        Input('sharpen-btn', 'n_clicks'),
        Input('invert-btn', 'n_clicks')
    ],
    [
        State('crop-slider', 'value'),
        State('blur-slider', 'value'),
        State('threshold1-slider', 'value'),
        State('threshold2-slider', 'value')
    ]
)
def process_image(contents, crop_clicks, blur_clicks, edge_clicks, grayscale_clicks, sharpen_clicks, invert_clicks, crop_ratio, blur_kernel, threshold1, threshold2):
    global uploaded_image, processed_image

    ctx = dash.callback_context
    if not ctx.triggered:
        return None

    # Determine which input triggered the callback
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'upload-image':
        if contents:
            _, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            np_image = np.frombuffer(decoded, np.uint8)
            uploaded_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            processed_image = uploaded_image.copy()
            return html.Img(src=image_to_base64(uploaded_image), style={"width": "50%"})

    if trigger_id == 'crop-btn':
        if processed_image is not None:
            h, w = processed_image.shape[:2]
            y1, y2 = int(h * crop_ratio / 100), int(h * (1 - crop_ratio / 100))
            x1, x2 = int(w * crop_ratio / 100), int(w * (1 - crop_ratio / 100))
            if y2 > y1 and x2 > x1:
                cropped = processed_image[y1:y2, x1:x2]
                processed_image = cv2.resize(cropped, (w, h))

    if trigger_id == 'blur-btn':
        if processed_image is not None:
            blur_kernel = max(1, blur_kernel // 2 * 2 + 1)
            processed_image = cv2.GaussianBlur(processed_image, (blur_kernel, blur_kernel), 0)

    if trigger_id == 'edge-btn':
        if processed_image is not None:
            edges = cv2.Canny(processed_image, threshold1, threshold2)
            processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if trigger_id == 'grayscale-btn':
        if processed_image is not None:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    if trigger_id == 'sharpen-btn':
        if processed_image is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_image = cv2.filter2D(processed_image, -1, kernel)

    if trigger_id == 'invert-btn':
        if processed_image is not None:
            processed_image = cv2.bitwise_not(processed_image)

    return html.Img(src=image_to_base64(processed_image), style={"width": "50%"})

# Callback to download the processed image
@app.callback(
    Output('download-link', 'href'),
    Input('download-link', 'n_clicks'),
    prevent_initial_call=True
)
def download_image(n_clicks):
    global processed_image
    if processed_image is not None:
        _, buffer = cv2.imencode('.jpg', processed_image)
        encoded_image = base64.b64encode(buffer).decode()
        return f'data:image/jpg;base64,{encoded_image}'
    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
