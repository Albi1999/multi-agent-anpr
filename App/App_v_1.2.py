from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.filechooser import FileChooserIconView
import cv2
import numpy as np

class LicensePlateApp(App):
    def build(self):
        # Main layout
        self.layout = BoxLayout(orientation="vertical")

        # Load Image Button
        self.load_btn = Button(text="Load Image")
        self.load_btn.bind(on_release=self.load_image)
        self.layout.add_widget(self.load_btn)

        # Image Display
        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)

        # Transformation Selection
        self.transformation_layout = BoxLayout(orientation="horizontal", size_hint_y=0.2)
        self.transformation_layout.visible = False

        self.crop_btn = Button(text="Crop and Resize")
        self.crop_btn.bind(on_release=self.crop_and_resize)
        self.transformation_layout.add_widget(self.crop_btn)

        self.grayscale_btn = Button(text="Grayscale")
        self.grayscale_btn.bind(on_release=lambda x: self.apply_transformation("grayscale"))
        self.transformation_layout.add_widget(self.grayscale_btn)

        self.edge_btn = Button(text="Edge Detection")
        self.edge_btn.bind(on_release=lambda x: self.apply_transformation("edge"))
        self.transformation_layout.add_widget(self.edge_btn)

        self.layout.add_widget(self.transformation_layout)

        # Slider and Label for Adjustment
        self.slider_label = Label(text="Adjust Parameter")
        self.layout.add_widget(self.slider_label)

        self.slider = Slider(min=0, max=100, value=50)
        self.slider.bind(value=self.update_slider)
        self.layout.add_widget(self.slider)

        # Navigation Buttons
        self.nav_layout = BoxLayout(orientation="horizontal", size_hint_y=0.1)

        self.prev_btn = Button(text="Previous")
        self.prev_btn.bind(on_release=self.previous_step)
        self.nav_layout.add_widget(self.prev_btn)

        self.home_btn = Button(text="Home")
        self.home_btn.bind(on_release=self.go_to_home)
        self.nav_layout.add_widget(self.home_btn)

        self.next_btn = Button(text="Next")
        self.next_btn.bind(on_release=self.next_step)
        self.nav_layout.add_widget(self.next_btn)

        self.layout.add_widget(self.nav_layout)

        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.history = []  # To manage undo
        self.redo_stack = []  # To manage redo

        return self.layout

    def load_image(self, instance):
        # File chooser to load an image
        filechooser = FileChooserIconView()
        filechooser.bind(on_submit=self.on_file_select)
        self.layout.add_widget(filechooser)

    def on_file_select(self, instance, selection, touch=None):
        if selection:
            file_path = selection[0]
            self.original_image = cv2.imread(file_path)
            self.processed_image = self.original_image.copy()
            self.history = [self.processed_image.copy()]
            self.redo_stack = []
            self.display_image(self.original_image)
            instance.parent.remove_widget(instance)  # Remove file chooser

    def display_image(self, image):
        # Convert image to Kivy-compatible texture
        if image is not None:
            buffer = cv2.flip(image, 0).tostring()
            texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture

    def update_slider(self, instance, value):
        # Apply adjustment based on slider value (example: brightness adjustment)
        if self.original_image is not None:
            alpha = float(value) / 50  # Scale value
            adjusted = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=0)
            self.processed_image = adjusted
            self.display_image(self.processed_image)

    def crop_and_resize(self, instance):
        # Crop and resize functionality
        if self.processed_image is not None:
            h, w = self.processed_image.shape[:2]
            crop = self.processed_image[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
            self.processed_image = cv2.resize(crop, (w, h))
            self.history.append(self.processed_image.copy())
            self.redo_stack.clear()
            self.display_image(self.processed_image)

    def apply_transformation(self, transformation):
        # Apply specific transformation
        if self.processed_image is not None:
            if transformation == "grayscale":
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
            elif transformation == "edge":
                self.processed_image = cv2.Canny(self.processed_image, 100, 200)
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
            self.history.append(self.processed_image.copy())
            self.redo_stack.clear()
            self.display_image(self.processed_image)

    def previous_step(self, instance):
        if len(self.history) > 1:
            self.redo_stack.append(self.history.pop())
            self.processed_image = self.history[-1].copy()
            self.display_image(self.processed_image)

    def next_step(self, instance):
        if self.redo_stack:
            self.processed_image = self.redo_stack.pop()
            self.history.append(self.processed_image.copy())
            self.display_image(self.processed_image)

    def go_to_home(self, instance):
        # Return to the transformation selection layout
        self.display_image(self.original_image)
        self.processed_image = self.original_image.copy()
        self.history = [self.original_image.copy()]
        self.redo_stack.clear()

if __name__ == "__main__":
    LicensePlateApp().run()
