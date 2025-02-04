import dash
from dash import dcc, html, Input, Output, State
import base64
from io import BytesIO
from PIL import Image
from Agent import LicensePlateAgent  # Import the agent class
import os

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Global variables to store images and agent
uploaded_image = None
agent = None

def initialize_agent(image_path):
    global agent
    agent = LicensePlateAgent(image_path)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Header function with an optional back button.
def get_header(show_back=True):
    return html.Div(
        style={
            "padding": "20px",
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "backgroundColor": "#333333"
        },
        children=[
            # Conditionally display the back button if show_back is True;
            # otherwise, use an empty div.
            html.A(
                "Back to Upload",
                href="/" if show_back else "#",
                style={
                    "color": "#fff",
                    "textDecoration": "none",
                    "backgroundColor": "#007bff",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "display": "inline-block"
                }
            ) if show_back else html.Div(),
            html.Div([
                html.H1("Interactive Image Processing", style={"color": "#ffcc00", "margin": "0"}),
                html.P("Upload and process images with real-time adjustments.", style={"margin": "0", "textAlign": "center"})
            ]),
            # Right-side container (currently empty)
            html.Div()
        ]
    )

def get_footer():
    return html.Div(
        style={
            "padding": "10px",
            "textAlign": "center",
            "backgroundColor": "#333333",
            "color": "#fff"
        },
        children=html.P("Developed by Alberto Calabrese, Marlon Helbing and Daniele Virzì, 2025", style={"margin": 0})
    )

# Upload page layout: header without back button, central upload area, and footer.
upload_layout = html.Div(
    style={
        "backgroundColor": "#1f1f1f",
        "color": "#f0f0f0",
        "fontFamily": "Arial",
        "minHeight": "100vh",
        "display": "flex",
        "flexDirection": "column"
    },
    children=[
        get_header(show_back=False),
        html.Div(
            style={
                "flex": "1",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
                "alignItems": "center",
                "textAlign": "center"
            },
            children=[
                html.H1("Upload Your Image", style={"color": "#ffcc00", "alignItems": "center", "justifyContent": "center"}),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div('Drag and drop or click to upload an image'),
                    style={
                        "width": "400px",
                        "height": "100px",
                        "padding": "10px",
                        "lineHeight": "100px",
                        "color": "#fff",
                        "backgroundColor": "#007bff",
                        "borderWidth": "2px",
                        "border": "none",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "cursor": "pointer"
                    },
                    multiple=False
                )
            ]
        ),
        get_footer()
    ]
)

# Main page layout: header with back button, sidebar controls, image preview area, and footer.
main_layout = html.Div(
    style={
        "backgroundColor": "#1f1f1f",
        "color": "#f0f0f0",
        "fontFamily": "Arial",
        "minHeight": "100vh"
    },
    children=[
        get_header(show_back=True),
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "padding": "20px", "flex": "1"},
            children=[
                # Sidebar for processing controls
                html.Div(
                    style={
                        "width": "30%",
                        "backgroundColor": "#2a2a2a",
                        "padding": "15px",
                        "borderRadius": "10px"
                    },
                    children=[
                        html.H2("Controls", style={"textAlign": "center", "color": "#fff"}),
                        html.Button(
                            'Auto Crop',
                            id='auto-crop-btn',
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "backgroundColor": "#28a745",
                                "color": "#fff",
                                "border": "none",
                                "borderRadius": "5px"
                            }
                        ),
                        html.Div(
                            style={
                                "marginTop": "20px",
                                "padding": "10px",
                                "backgroundColor": "#3a3a3a",
                                "borderRadius": "10px"
                            },
                            children=[
                                html.Button(
                                    'Exposure',
                                    id='exposure-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "49%",
                                        "padding": "10px",
                                        "backgroundColor": "#ff8700",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginRight": "2%"
                                    }
                                ),
                                html.Button(
                                    'Contrast',
                                    id='contrast-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "49%",
                                        "padding": "10px",
                                        "backgroundColor": "#8a00dc",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px"
                                    }
                                ),
                                html.Button(
                                    'Brightness',
                                    id='brightness-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "49%",
                                        "padding": "10px",
                                        "backgroundColor": "#dc3545",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginTop": "10px",
                                        "marginRight": "2%"
                                    }
                                ),
                                html.Button(
                                    'Sharpness',
                                    id='sharpen-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "49%",
                                        "padding": "10px",
                                        "backgroundColor": "#007bff",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginTop": "10px",
                                        "marginBottom": "50px"
                                    }
                                ),
                                dcc.Slider(
                                    id='adjustment-slider',
                                    min=-1,
                                    max=1,
                                    step=0.1,
                                    value=0,
                                    marks={-1: "-1", 0: "0", 1: "1"},
                                    tooltip={"always_visible": True}
                                )
                            ]
                        ),
                        html.Div(
                            style={
                                "marginTop": "20px",
                                "padding": "10px",
                                "backgroundColor": "#3a3a3a",
                                "borderRadius": "10px"
                            },
                            children=[
                                html.Button(
                                    'Edge Detection',
                                    id='edge-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "backgroundColor": "#17a2b8",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px"
                                    }
                                ),
                                html.Button(
                                    'Grayscale',
                                    id='grayscale-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "backgroundColor": "#6c757d",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginTop": "10px"
                                    }
                                ),
                                html.Button(
                                    'Invert Colors',
                                    id='invert-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "backgroundColor": "#8a00dc",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginTop": "10px"
                                    }
                                ),
                                html.Button(
                                    'Perspective Correction',
                                    id='persp-corr-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "backgroundColor": "#28a745",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginTop": "10px"
                                    }
                                ),
                                html.Button(
                                    'Threshold',
                                    id='threshold-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "backgroundColor": "#ffc107",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginTop": "10px"
                                    }
                                )
                            ]
                        )
                    ]
                ),
                # Image preview area with undo, redo, reset, and download
                html.Div(
                    style={
                        "width": "65%",
                        "backgroundColor": "#2a2a2a",
                        "padding": "15px",
                        "borderRadius": "10px",
                        "textAlign": "center"
                    },
                    children=[
                        html.H2("Image Preview", style={"textAlign": "center", "color": "#fff"}),
                        html.Div(id='image-display', style={"marginTop": "20px"}),
                        html.Div(
                            style={
                                "marginTop": "20px",
                                "display": "flex",
                                "justifyContent": "center",
                                "gap": "10px"
                            },
                            children=[
                                html.Button(
                                    'Undo',
                                    id='undo-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "backgroundColor": "#6c757d",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px"
                                    }
                                ),
                                html.Button(
                                    'Redo',
                                    id='redo-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "backgroundColor": "#ffc107",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px"
                                    }
                                ),
                                html.Button(
                                    'Reset',
                                    id='reset-btn',
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "backgroundColor": "#dc3545",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "5px"
                                    }
                                )
                            ]
                        ),
                        html.A(
                            "Download Processed Image",
                            id="download-link",
                            download="processed_image.png",
                            href="",
                            target="_blank",
                            style={
                                "display": "block",
                                "textAlign": "center",
                                "marginTop": "20px",
                                "padding": "10px",
                                "backgroundColor": "#007bff",
                                "color": "#fff",
                                "border": "none",
                                "borderRadius": "5px",
                                "textDecoration": "none"
                            }
                        )
                    ]
                )
            ]
        ),
        get_footer()
    ]
)

# Top-level layout with dcc.Location (using refresh=True for full page reload on URL change)
app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
])

# Callback to process the uploaded image and redirect to the main page.
@app.callback(
    Output('url', 'href'),
    Input('upload-image', 'contents'),
    prevent_initial_call=True
)

def redirect_on_upload(uploaded_content):
    global uploaded_image  # add global declaration
    if uploaded_content is not None:
        content_type, content_string = uploaded_content.split(',')
        decoded = base64.b64decode(content_string)
        image_path = "uploaded_image.png"
        with open(image_path, "wb") as f:
            f.write(decoded)
        # Save the original image to the global variable
        uploaded_image = Image.open(BytesIO(decoded))
        initialize_agent(image_path)
        return '/main'
    return dash.no_update

# Routing callback to display the appropriate page.
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/main':
        return main_layout
    return upload_layout

# Callback to process image adjustments and update the preview/download link.
@app.callback(
    [Output('image-display', 'children'),
     Output('download-link', 'href')],
    [Input('url', 'pathname'),
     Input('auto-crop-btn', 'n_clicks'),
     Input('persp-corr-btn', 'n_clicks'),
     Input('edge-btn', 'n_clicks'),
     Input('exposure-btn', 'n_clicks'),
     Input('contrast-btn', 'n_clicks'),
     Input('brightness-btn', 'n_clicks'),
     Input('threshold-btn', 'n_clicks'),
     Input('grayscale-btn', 'n_clicks'),
     Input('sharpen-btn', 'n_clicks'),
     Input('invert-btn', 'n_clicks'),
     Input('undo-btn', 'n_clicks'),
     Input('redo-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks')],
    State('adjustment-slider', 'value'),
    prevent_initial_call=False
)
def process_image(pathname, auto_crop_clicks, persp_corr_clicks, edge_clicks,
                  exposure_clicks, contrast_clicks, brightness_clicks, threshold_clicks,
                  grayscale_clicks, sharpen_clicks, invert_clicks, undo_clicks, redo_clicks,
                  reset_clicks, slider_value):
    global uploaded_image, agent
    if not agent:
        return dash.no_update, dash.no_update

    ctx = dash.callback_context
    
    # When the URL change triggers the callback, display the original uploaded image.
    if any(trigger['prop_id'].startswith('url') for trigger in ctx.triggered):
        encoded = image_to_base64(uploaded_image)
        image_element = html.Img(
            src=f"data:image/png;base64,{encoded}",
            style={'width': '90%', 'borderRadius': '10px'}
        )
        return image_element, f"data:image/png;base64,{encoded}"

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    factor = slider_value + 1  # Scale slider value from (-1,1) to (0,2)

    # For the reset button, restore the original uploaded image.
    if triggered_id == 'reset-btn':
        if uploaded_image:
            uploaded_image.save(agent.current_image_path)
            encoded = image_to_base64(uploaded_image)
            return html.Img(
                src=f"data:image/png;base64,{encoded}",
                style={'width': '90%', 'borderRadius': '10px'}
            ), ""

    processed_image_path = None
    if triggered_id == 'auto-crop-btn':
        processed_image_path = agent.auto_crop_tool()
    elif triggered_id == 'exposure-btn':
        processed_image_path = agent.adjust_exposure_tool(factor)
    elif triggered_id == 'contrast-btn':
        processed_image_path = agent.adjust_contrast_tool(factor)
    elif triggered_id == 'brightness-btn':
        processed_image_path = agent.adjust_brilliance_tool(factor)
    elif triggered_id == 'edge-btn':
        processed_image_path = agent.edge_detection_tool()
    elif triggered_id == 'grayscale-btn':
        processed_image_path = agent.grayscale_tool()
    elif triggered_id == 'sharpen-btn':
        processed_image_path = agent.adjust_sharpness_tool(factor)
    elif triggered_id == 'invert-btn':
        processed_image_path = agent.invert_tool()
    elif triggered_id == 'undo-btn':
        processed_image_path = agent.go_back()
    elif triggered_id == 'redo-btn':
        processed_image_path = agent.go_forward()

    if not processed_image_path:
        processed_image_path = agent.current_image_path

    if processed_image_path and os.path.exists(processed_image_path):
        processed_image = Image.open(processed_image_path)
        encoded_image = image_to_base64(processed_image)
        return html.Img(
            src=f"data:image/png;base64,{encoded_image}",
            style={'width': '90%', 'borderRadius': '10px'}
        ), f"data:image/png;base64,{encoded_image}"
    else:
        return html.Div("Error: Processed image not found.", style={"color": "red"}), dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
