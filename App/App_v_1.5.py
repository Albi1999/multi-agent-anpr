import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup

class ImageProcessing:
    """Handles all the image processing functions as in Agent.py"""

    @staticmethod
    def crop_and_resize(image, crop_ratio=0.1):
        h, w = image.shape[:2]
        y1, y2 = int(h * crop_ratio), int(h * (1 - crop_ratio))
        x1, x2 = int(w * crop_ratio), int(w * (1 - crop_ratio))
        
        # Validate cropping dimensions
        if y2 <= y1 or x2 <= x1:
            return image  # Return the original image if crop dimensions are invalid
        
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h))
        return resized

    @staticmethod
    def grayscale(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def edge_detection(image, threshold1, threshold2):
        edges = cv2.Canny(image, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def blur(image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def sharpen(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def invert_colors(image):
        return cv2.bitwise_not(image)

class LoadScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        layout = BoxLayout(orientation="vertical")
        load_btn = Button(text="Load Image")
        load_btn.bind(on_release=self.load_image)
        layout.add_widget(load_btn)
        self.add_widget(layout)

    def load_image(self, instance):
        filechooser = FileChooserIconView()
        popup = Popup(title="Choose Image", content=filechooser, size_hint=(0.9, 0.9))
        filechooser.bind(on_submit=lambda *args: self.on_file_select(popup, args))
        popup.open()

    def on_file_select(self, popup, args):
        selection = args[1]
        if selection:
            file_path = selection[0]
            self.app.original_image = cv2.imread(file_path)
            self.app.processed_image = self.app.original_image.copy()
            self.app.history = [self.app.processed_image.copy()]
            self.app.redo_stack = []
            self.manager.current = "home"
            self.manager.get_screen("home").update_image_display()
        popup.dismiss()

class HomeScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.layout = BoxLayout(orientation="vertical")

        # Image Display
        self.image_display = Image(size_hint_y=0.5)
        self.layout.add_widget(self.image_display)

        # Sliders and Labels
        self.slider_layout = BoxLayout(orientation="vertical", size_hint_y=0.3)

        self.label = Label(text="Adjust Parameters")
        self.slider_layout.add_widget(self.label)

        self.slider1 = Slider(min=0, max=255, value=50)
        self.slider2 = Slider(min=0, max=255, value=150)
        self.slider_layout.add_widget(self.slider1)
        self.slider_layout.add_widget(self.slider2)
        self.layout.add_widget(self.slider_layout)

        # Buttons
        btn_layout = BoxLayout(orientation="horizontal", size_hint_y=0.2)

        crop_btn = Button(text="Crop and Resize")
        crop_btn.bind(on_release=lambda x: self.apply_processing(ImageProcessing.crop_and_resize, crop_ratio=self.slider1.value / 100))
        btn_layout.add_widget(crop_btn)

        grayscale_btn = Button(text="Grayscale")
        grayscale_btn.bind(on_release=lambda x: self.apply_processing(ImageProcessing.grayscale))
        btn_layout.add_widget(grayscale_btn)

        edge_btn = Button(text="Edge Detection")
        edge_btn.bind(on_release=lambda x: self.apply_processing(ImageProcessing.edge_detection, threshold1=int(self.slider1.value), threshold2=int(self.slider2.value)))
        btn_layout.add_widget(edge_btn)

        blur_btn = Button(text="Blur")
        blur_btn.bind(on_release=lambda x: self.apply_processing(ImageProcessing.blur, kernel_size=int(self.slider1.value / 10) * 2 + 1))
        btn_layout.add_widget(blur_btn)

        sharpen_btn = Button(text="Sharpen")
        sharpen_btn.bind(on_release=lambda x: self.apply_processing(ImageProcessing.sharpen))
        btn_layout.add_widget(sharpen_btn)

        invert_btn = Button(text="Invert Colors")
        invert_btn.bind(on_release=lambda x: self.apply_processing(ImageProcessing.invert_colors))
        btn_layout.add_widget(invert_btn)

        self.layout.add_widget(btn_layout)

        # Save Button
        save_btn = Button(text="Save Image", size_hint_y=0.1)
        save_btn.bind(on_release=self.save_image)
        self.layout.add_widget(save_btn)

        self.add_widget(self.layout)

    def update_image_display(self):
        if self.app.processed_image is not None:
            image_rgb = cv2.cvtColor(self.app.processed_image, cv2.COLOR_BGR2RGB)
            buffer = cv2.flip(image_rgb, 0).tobytes()
            texture = Texture.create(size=(image_rgb.shape[1], image_rgb.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
            self.image_display.texture = texture

    def apply_processing(self, processing_func, **kwargs):
        if self.app.processed_image is not None:
            self.app.processed_image = processing_func(self.app.processed_image, **kwargs)
            self.app.history.append(self.app.processed_image.copy())
            self.app.redo_stack.clear()
            self.update_image_display()

    def save_image(self, instance):
        filechooser = FileChooserIconView()
        popup = Popup(title="Save Image", content=filechooser, size_hint=(0.9, 0.9))
        filechooser.bind(on_submit=lambda *args: self.do_save_image(popup, args))
        popup.open()

    def do_save_image(self, popup, args):
        selection = args[1]
        if selection:
            cv2.imwrite(selection[0], self.app.processed_image)
        popup.dismiss()

class LicensePlateApp(App):
    def build(self):
        self.original_image = None
        self.processed_image = None
        self.history = []
        self.redo_stack = []

        sm = ScreenManager()
        sm.add_widget(LoadScreen(app=self, name="load"))
        sm.add_widget(HomeScreen(app=self, name="home"))

        return sm

if __name__ == "__main__":
    LicensePlateApp().run()
