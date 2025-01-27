from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scatter import Scatter
from kivy.uix.popup import Popup
import cv2
import numpy as np

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
            home_screen = self.manager.get_screen("home")
            home_screen.update_image_display()
        popup.dismiss()

class HomeScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.layout = BoxLayout(orientation="vertical")

        # Image Display
        self.image_widget = Scatter(size_hint_y=0.7)
        self.image_display = Image()
        self.image_widget.add_widget(self.image_display)
        self.layout.add_widget(self.image_widget)

        # Functionality Buttons
        btn_layout = BoxLayout(orientation="horizontal", size_hint_y=0.2)

        crop_btn = Button(text="Crop and Resize")
        crop_btn.bind(on_release=lambda x: setattr(self.manager, 'current', 'crop'))
        btn_layout.add_widget(crop_btn)

        grayscale_btn = Button(text="Grayscale")
        grayscale_btn.bind(on_release=lambda x: setattr(self.manager, 'current', 'grayscale'))
        btn_layout.add_widget(grayscale_btn)

        edge_btn = Button(text="Edge Detection")
        edge_btn.bind(on_release=lambda x: setattr(self.manager, 'current', 'edge'))
        btn_layout.add_widget(edge_btn)

        self.layout.add_widget(btn_layout)

        # Save and Zoom Buttons
        utility_layout = BoxLayout(orientation="horizontal", size_hint_y=0.1)

        save_btn = Button(text="Save Image")
        save_btn.bind(on_release=self.save_image)
        utility_layout.add_widget(save_btn)

        zoom_btn = Button(text="Zoom Image")
        zoom_btn.bind(on_release=self.zoom_image)
        utility_layout.add_widget(zoom_btn)

        self.layout.add_widget(utility_layout)

        self.add_widget(self.layout)

    def update_image_display(self):
        if self.app.processed_image is not None:
            image_rgb = cv2.cvtColor(self.app.processed_image, cv2.COLOR_BGR2RGB)
            buffer = cv2.flip(image_rgb, 0).tobytes()
            texture = Texture.create(size=(image_rgb.shape[1], image_rgb.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
            self.image_display.texture = texture

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

    def zoom_image(self, instance):
        if self.app.processed_image is not None:
            h, w = self.app.processed_image.shape[:2]
            self.image_widget.scale = min(self.width / w, self.height / h)

class CropScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.layout = BoxLayout(orientation="vertical")

        # Navigation Buttons
        nav_layout = BoxLayout(orientation="horizontal", size_hint_y=0.1)

        back_btn = Button(text="Back")
        back_btn.bind(on_release=lambda x: setattr(self.manager, 'current', 'home'))
        nav_layout.add_widget(back_btn)

        apply_btn = Button(text="Apply")
        apply_btn.bind(on_release=self.apply_crop)
        nav_layout.add_widget(apply_btn)

        self.layout.add_widget(nav_layout)
        self.add_widget(self.layout)

    def apply_crop(self, instance):
        if self.app.processed_image is not None:
            h, w = self.app.processed_image.shape[:2]
            crop = self.app.processed_image[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
            self.app.processed_image = cv2.resize(crop, (w, h))
            self.app.history.append(self.app.processed_image.copy())
            self.app.redo_stack.clear()
            self.manager.current = "home"
            self.manager.get_screen("home").update_image_display()

class GrayscaleScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.layout = BoxLayout(orientation="vertical")

        # Image Display
        self.image_widget = Image(size_hint_y=0.7)
        self.layout.add_widget(self.image_widget)

        # Slider for effect intensity
        slider = Slider(min=0, max=100, value=50)
        slider.bind(value=self.apply_grayscale)
        self.layout.add_widget(slider)

        # Navigation Buttons
        nav_layout = BoxLayout(orientation="horizontal", size_hint_y=0.1)

        back_btn = Button(text="Back")
        back_btn.bind(on_release=lambda x: setattr(self.manager, 'current', 'home'))
        nav_layout.add_widget(back_btn)

        self.layout.add_widget(nav_layout)
        self.add_widget(self.layout)

    def apply_grayscale(self, instance, value):
        if self.app.processed_image is not None:
            gray = cv2.cvtColor(self.app.processed_image, cv2.COLOR_BGR2GRAY)
            self.app.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.app.history.append(self.app.processed_image.copy())
            self.app.redo_stack.clear()
            self.manager.get_screen("home").update_image_display()
            self.update_image_display()

    def update_image_display(self):
        if self.app.processed_image is not None:
            image_rgb = cv2.cvtColor(self.app.processed_image, cv2.COLOR_BGR2RGB)
            buffer = cv2.flip(image_rgb, 0).tobytes()
            texture = Texture.create(size=(image_rgb.shape[1], image_rgb.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
            self.image_widget.texture = texture

class EdgeDetectionScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.layout = BoxLayout(orientation="vertical")

        # Image Display
        self.image_widget = Image(size_hint_y=0.7)
        self.layout.add_widget(self.image_widget)

        # Slider for edge threshold
        slider = Slider(min=0, max=255, value=100)
        slider.bind(value=self.apply_edge_detection)
        self.layout.add_widget(slider)

        # Navigation Buttons
        nav_layout = BoxLayout(orientation="horizontal", size_hint_y=0.1)

        back_btn = Button(text="Back")
        back_btn.bind(on_release=lambda x: setattr(self.manager, 'current', 'home'))
        nav_layout.add_widget(back_btn)

        self.layout.add_widget(nav_layout)
        self.add_widget(self.layout)

    def apply_edge_detection(self, instance, value):
        if self.app.processed_image is not None:
            edges = cv2.Canny(self.app.processed_image, threshold1=value, threshold2=value*2)
            self.app.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.app.history.append(self.app.processed_image.copy())
            self.app.redo_stack.clear()
            self.manager.get_screen("home").update_image_display()
            self.update_image_display()

    def update_image_display(self):
        if self.app.processed_image is not None:
            image_rgb = cv2.cvtColor(self.app.processed_image, cv2.COLOR_BGR2RGB)
            buffer = cv2.flip(image_rgb, 0).tobytes()
            texture = Texture.create(size=(image_rgb.shape[1], image_rgb.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
            self.image_widget.texture = texture

class LicensePlateApp(App):
    def build(self):
        self.original_image = None
        self.processed_image = None
        self.history = []
        self.redo_stack = []

        sm = ScreenManager()
        sm.add_widget(LoadScreen(app=self, name="load"))
        sm.add_widget(HomeScreen(app=self, name="home"))
        sm.add_widget(CropScreen(app=self, name="crop"))
        sm.add_widget(GrayscaleScreen(app=self, name="grayscale"))
        sm.add_widget(EdgeDetectionScreen(app=self, name="edge"))

        return sm

if __name__ == "__main__":
    LicensePlateApp().run()
