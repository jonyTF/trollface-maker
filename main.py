import kivy
kivy.require('1.10.1')

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.anchorlayout import AnchorLayout

class StartPage(AnchorLayout):
    pass

class TrollfaceMakerApp(App):
    def build(self):
        #layout = AnchorLayout(anchor_x='left', anchor_y='top')
        #btn = Button(text='lol', size_hint=(.2,.2))
        #layout.add_widget(btn)
        #return layout
        return StartPage()

if __name__ == '__main__':
    TrollfaceMakerApp().run()