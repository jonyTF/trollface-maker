import kivy
kivy.require('1.10.1')

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

import random

class BackgroundImage(Image):
    def __init__(self, **kwargs):
        super(BackgroundImage, self).__init__()

    def texture_width(self):
        return self.texture.size[0]

    def texture_height(self):
        return self.texture.size[1]

    def rescale(self, width, height):
        """
        Resize the image to fit the given dimensions, zooming in or out if
        needed without losing the aspect ratio
        :param width: target width
        :param height: target height
        :return: new dimensions as a tuple (width, height)
        """
        ratio = 0.0
        new_width = 0.0
        new_height = 0.0

        target_width = float(width)
        target_height = float(height)

        image_width = float(self.texture_width())
        image_height = float(self.texture_height())

        ratio = target_width / image_width
        new_width = image_width * ratio
        new_height = image_height * ratio

        if (new_height < target_height):
            ratio = target_height / new_height
            new_height *= ratio
            new_width *= ratio

        if new_width > 0 and new_height > 0:
            self.width = new_width
            self.height = new_height

        return (new_width, new_height)

    def get_image(self):
        if random.random() < .5:
            return 'img/trump_troll.jpg'
        return 'img/trolla_lisa.jpg'

class StartScreen(Screen):
    def take_photo(self):
        print('take a photo')
    
    def choose_photo(self):
        print('choose a photo')

    def open_gallery(self):
        print('Open the gallery!')

class Manager(ScreenManager):
    pass

class TrollfaceMakerApp(App):
    def build(self):
        #layout = AnchorLayout(anchor_x='left', anchor_y='top')
        #btn = Button(text='lol', size_hint=(.2,.2))
        #layout.add_widget(btn)
        #return layout
        return Manager()

if __name__ == '__main__':
    TrollfaceMakerApp().run()