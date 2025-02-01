import dash
from dash import dcc, html, Input, Output, State
import base64
from io import BytesIO
from PIL import Image
from Agent import LicensePlateAgent  # Import the agent class

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variables to store images and history
uploaded_image = None
agent = None  # Will be initialized when an image is uploaded

# Function to initialize the agent with the uploaded image
def initialize_agent(image_path):
    global agent
    agent = LicensePlateAgent(image_path)

# Convert image to base64 for display in Dash
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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

                        html.Button('Perspective Correction', id='persp-corr-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#28a745", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "20px"}),

                        html.Button('Blur', id='blur-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#17a2b8", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Adjust Exposure', id='exposure-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#ff8700", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Adjust Contrast', id='contrast-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#8a00dc", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Adjust Brightness', id='brightness-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#dc3545", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Threshold', id='threshold-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Grayscale', id='grayscale-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#6c757d", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Sharpen', id='sharpen-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#007bff", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Invert Colors', id='invert-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#28a745", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

                        html.Button('Undo', id='undo-btn', n_clicks=0, 
                                  style={"width": "49%", "padding": "10px", "backgroundColor": "#6c757d", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px", "marginRight": "2%"}),

                        html.Button('Redo', id='redo-btn', n_clicks=0, 
                                  style={"width": "49%", "padding": "10px", "backgroundColor": "#ffc107", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),
                        
                        html.Button('Reset', id='reset-btn', n_clicks=0, 
                                  style={"width": "100%", "padding": "10px", "backgroundColor": "#dc3545", "color": "#fff", "border": "none", "borderRadius": "5px", "marginTop": "10px"}),

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


@app.callback(
    [Output('image-display', 'children'),
     Output('download-link', 'href')],
    [Input('upload-image', 'contents'),
     Input('persp-corr-btn', 'n_clicks'), Input('blur-btn', 'n_clicks'),
     Input('exposure-btn', 'n_clicks'), Input('contrast-btn', 'n_clicks'),
     Input('brightness-btn', 'n_clicks'), Input('threshold-btn', 'n_clicks'),
     Input('grayscale-btn', 'n_clicks'), Input('sharpen-btn', 'n_clicks'),
     Input('invert-btn', 'n_clicks'), Input('undo-btn', 'n_clicks'),
     Input('redo-btn', 'n_clicks'), Input('reset-btn', 'n_clicks')],
    prevent_initial_call=True
)

def handle_image_processing(uploaded_content, auto_crop_clicks, blur_clicks, exposure_clicks, contrast_clicks, 
                            brightness_clicks, threshold_clicks, grayscale_clicks, sharpen_clicks, invert_clicks, 
                            undo_clicks, redo_clicks, reset_clicks):
    global uploaded_image, agent

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'upload-image':
        if uploaded_content:
            content_type, content_string = uploaded_content.split(',')
            decoded = base64.b64decode(content_string)
            image_path = "uploaded_image.png"
            with open(image_path, "wb") as f:
                f.write(decoded)
            uploaded_image = Image.open(BytesIO(decoded))
            initialize_agent(image_path)
            return html.Img(src=uploaded_content, style={'width': '90%', 'borderRadius': '10px'}), dash.no_update

    if not agent:
        return dash.no_update, dash.no_update

    if button_id == 'reset-btn':
        if uploaded_image:
            uploaded_image.save(agent.current_image_path)
            return html.Img(src=f"data:image/png;base64,{image_to_base64(uploaded_image)}", style={'width': '90%', 'borderRadius': '10px'}), ""

    if button_id == 'persp-corr-btn':
        processed_image_path = agent.perspective_correction()
    elif button_id == 'blur-btn':
        processed_image_path = agent.edge_detection_tool()
    elif button_id == 'exposure-btn':
        processed_image_path = agent.adjust_exposure_tool(1.2)
    elif button_id == 'contrast-btn':
        processed_image_path = agent.adjust_contrast_tool(1.5)
    elif button_id == 'brightness-btn':
        processed_image_path = agent.adjust_brilliance_tool(1.1)
    elif button_id == 'threshold-btn':
        processed_image_path = agent.adaptive_thresholding()
    elif button_id == 'grayscale-btn':
        processed_image_path = agent.grayscale_tool()
    elif button_id == 'sharpen-btn':
        processed_image_path = agent.adjust_sharpness_tool(2.0)
    elif button_id == 'invert-btn':
        # No specific method for invert; use a custom implementation
        img = Image.open(agent.current_image_path)
        inverted_img = Image.eval(img, lambda x: 255 - x)
        inverted_img.save(agent.current_image_path)
        processed_image_path = agent.current_image_path
    elif button_id == 'undo-btn':
        processed_image_path = agent.go_back()
    elif button_id == 'redo-btn':
        processed_image_path = agent.go_forward()

    if processed_image_path:
        processed_image = Image.open(processed_image_path)
        encoded_image = image_to_base64(processed_image)
        download_href = f"data:image/png;base64,{encoded_image}"
        return html.Img(src=f"data:image/png;base64,{encoded_image}", style={'width': '90%', 'borderRadius': '10px'}), download_href

    return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
