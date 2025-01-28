import dash
from dash import dcc, html, Input, Output, State
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Interactive Image Processing"),
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload Image'),
        multiple=False
    ),
    html.Div(id='image-display'),
    html.Div([
        dcc.Slider(id='crop-slider', min=0, max=50, step=1, value=10, marks={i: f'{i}%' for i in range(0, 51, 10)}),
        html.Button('Crop & Resize', id='crop-btn', n_clicks=0),
    ]),
    html.Div([
        dcc.Slider(id='threshold-slider', min=0, max=255, step=1, value=100, marks={i: f'{i}' for i in range(0, 256, 50)}),
        html.Button('Edge Detection', id='edge-btn', n_clicks=0),
    ]),
    html.Button('Save Processed Image', id='save-btn', n_clicks=0),
    html.A(id='download-link', children="Click here to download", style={'display': 'none'})
])

# Callbacks
@app.callback(
    Output('image-display', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def display_image(contents, filename):
    if contents:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        np_img = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        cv2.imwrite('uploaded_image.jpg', img)
        _, buffer = cv2.imencode('.jpg', img)
        encoded_img = base64.b64encode(buffer).decode()
        return html.Img(src=f'data:image/jpg;base64,{encoded_img}', style={'width': '50%'})

@app.callback(
    Output('image-display', 'children', allow_duplicate=True),
    Input('crop-btn', 'n_clicks'),
    State('crop-slider', 'value'),
    prevent_initial_call=True
)
def crop_and_resize(n_clicks, crop_ratio):
    if n_clicks > 0:
        img = cv2.imread('uploaded_image.jpg')
        h, w = img.shape[:2]
        y1, y2 = int(h * crop_ratio / 100), int(h * (1 - crop_ratio / 100))
        x1, x2 = int(w * crop_ratio / 100), int(w * (1 - crop_ratio / 100))
        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h))
        cv2.imwrite('processed_image.jpg', resized)
        _, buffer = cv2.imencode('.jpg', resized)
        encoded_img = base64.b64encode(buffer).decode()
        return html.Img(src=f'data:image/jpg;base64,{encoded_img}', style={'width': '50%'})

@app.callback(
    Output('image-display', 'children', allow_duplicate=True),
    Input('edge-btn', 'n_clicks'),
    State('threshold-slider', 'value'),
    prevent_initial_call=True
)
def edge_detection(n_clicks, threshold):
    if n_clicks > 0:
        img = cv2.imread('uploaded_image.jpg', cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, threshold1=threshold, threshold2=threshold * 2)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.imwrite('processed_image.jpg', edges)
        _, buffer = cv2.imencode('.jpg', edges)
        encoded_img = base64.b64encode(buffer).decode()
        return html.Img(src=f'data:image/jpg;base64,{encoded_img}', style={'width': '50%'})

@app.callback(
    Output('download-link', 'href'),
    Input('save-btn', 'n_clicks'),
    prevent_initial_call=True
)
def save_image(n_clicks):
    if n_clicks > 0:
        with open('processed_image.jpg', 'rb') as file:
            encoded = base64.b64encode(file.read()).decode()
        return f'data:image/jpg;base64,{encoded}'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
