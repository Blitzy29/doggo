import pygame
import time


def play_music(music='spam-x3.mp3'):

    pygame.mixer.init()
    pygame.mixer.music.load("data/external/"+music)
    pygame.mixer.music.play()
    time.sleep(20)
