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

        self.next_btn = Button(text="Next")
        self.next_btn.bind(on_release=self.next_step)
        self.nav_layout.add_widget(self.next_btn)

        self.layout.add_widget(self.nav_layout)

        # Initialize variables
        self.original_image = None
        self.processed_image = None

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

    def previous_step(self, instance):
        # Placeholder for navigating to the previous step
        print("Previous step functionality not implemented yet.")

    def next_step(self, instance):
        # Placeholder for navigating to the next step
        print("Next step functionality not implemented yet.")

if __name__ == "__main__":
    LicensePlateApp().run()
