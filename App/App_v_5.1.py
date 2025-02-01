import dash
from dash import dcc, html, Input, Output, State
import base64
from io import BytesIO
from PIL import Image
from Agent import LicensePlateAgent  # Import the agent class
import cv2
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variables
agent = None  # Will be initialized when an image is uploaded
uploaded_image = None
processed_image = None
history = []
redo_stack = []

# Function to initialize the agent with the uploaded image
def initialize_agent(image_path):
    global agent
    agent = LicensePlateAgent(image_path)

# Convert image to base64 for display in Dash
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#28a745", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "20px"}),

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
                        html.Button('Grayscale', id='grayscale-btn', n_clicks=0, 
                                  style={"width": "32%", "padding": "10px", "backgroundColor": "#6c757d", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "20px", "marginRight": "2%"}),
                        html.Button('Sharpen', id='sharpen-btn', n_clicks=0, 
                                  style={"width": "32%", "padding": "10px", "backgroundColor": "#007bff", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px", "marginRight": "2%"}),
                        html.Button('Invert Colors', id='invert-btn', n_clicks=0, 
                                  style={"width": "32%", "padding": "10px", "backgroundColor": "#dc3545", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),
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

@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    prevent_initial_call=True
)
def upload_image(contents):
    global uploaded_image
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image_path = "uploaded_image.png"
        with open(image_path, "wb") as f:
            f.write(decoded)
        uploaded_image = Image.open(BytesIO(decoded))
        initialize_agent(image_path)  # Initialize the agent with the image
        return html.Img(src=contents, style={'width': '50%'})

@app.callback(
    Output('processed-image', 'src'),
    Output('processed-image', 'style'),
    [Input('edge-detection-button', 'n_clicks'),
     Input('adjust-exposure-button', 'n_clicks'),
     Input('adjust-contrast-button', 'n_clicks'),
     Input('adaptive-thresholding-button', 'n_clicks')],
    prevent_initial_call=True
)
def process_image(edge_clicks, exposure_clicks, contrast_clicks, threshold_clicks):
    global processed_image
    ctx = dash.callback_context
    if not ctx.triggered or not agent or not uploaded_image:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'edge-detection-button':
        processed_image = agent.edge_detection()
    elif button_id == 'adjust-exposure-button':
        processed_image = agent.adjust_exposure()
    elif button_id == 'adjust-contrast-button':
        processed_image = agent.adjust_contrast()
    elif button_id == 'adaptive-thresholding-button':
        processed_image = agent.adaptive_thresholding()
    
    processed_image_pil = Image.fromarray(processed_image)
    encoded_image = image_to_base64(processed_image_pil)
    return f'data:image/png;base64,{encoded_image}', {'display': 'block'}

if __name__ == '__main__':
    app.run_server(debug=True)
