import pygame as pg
import numpy as np
from random import randint, shuffle
import os
from shutil import rmtree
import pickle as pkl
import pandas as pd

pg.init()

SAVE_TRAINING_FOLDER = 'trainings/' # pasta onde serão salvos os treinamentos
SAVE_NN_FOLDER = 'saves/neural_networks/' # pasta onde serão salvas as redes neurais
SAVE_EVOLUTION_FOLDER = 'saves/evolutions/' # pasta onde serão salvas as redes neurais

NN_LAYOUT = [14, 7, 4] # formato da rede neural - [10, 10, 7, 4]

#==== DEFAULT:
SIZE_POPULATION = 500 # tamanho de cada população
ETA = 10 # taxa de variação para cada valor genético
SCALE = 50 # escala de tamanho do jogo
LIFES = 1 # chances de cada cobrinha
APPLE_SCORE = 1000 # pontos a serem adicionados se a cobrinha pegar maçã
STEP_LIM = 300 # limites de passos para morrer

BACKGROUND_TRAINING_COLOR = (190,200,195) # cor do fundo de treinamento
BACKGROUND_GAME_COLOR = (40,20,60) # cor do fundo do jogo
SNAKE_COLOR = (250,250,250) # cor da cobrinha
APPLE_COLOR = (255,0,0) # cor da maçã

PLT_COLORS = [
    (150,255,200), # médias das pontuações
    (255,150,70), # melhores pontuações
    (0,150,100), # média móvel das medias das pontuações
    (255,0,0) # média móvel das melhores pontuações
]

def moving_average(a, w): # média móvel
    y = []
    f = lambda x: x if x > 0 else 0
    for i in nrange(len(a)):
        dy = a[f(i-w//2):f(i+w//2)]
        if len(dy) < w: y.append(sum(dy)/len(dy))
        else: y.append(sum(dy)/w)
    return y

def nrange(x0 ,xi=None, dx=1): # contagem
    if not xi: xi = x0; x0 = 0
    while x0 < xi:
        yield x0
        x0 += dx

def rmap(h, hmin, hmax, ymin, ymax):
    return (ymax-ymin)/((hmax-hmin) if (hmax-hmin) else 1)*(h - hmin) + ymin # mapear valor

class LineData:
    def __init__(self, *args): self.points, self.style, self.color, self.width = args

class Graph: # gráfico para plotar dados
    def __init__(self, position, size):
        self.position = position
        self.width, self.height = size
        self.bg_color = [255,255,255]
        self.surface = pg.Surface(size)
        self.data = []
        self.settings = {'x_max':None, 'x_min':None, 'y_max':None, 'y_min':None}
        self.clear()

        self.axis_font = pg.font.SysFont('calibre', 18)
        self.axis_color = (180,180,180)

        self.xymouse = False
        self.xymouse_surface = pg.Surface(size, pg.SRCALPHA)
        self.xymouse_color = (0,100,100)
        self.grid = False

        self.origin = np.array([0, self.height])
        self.horientation = np.array([1, -1])
        (self.x_max, self.x_min), (self.y_max, self.y_min) = (1, -1), (1, -1)
        self.scale = np.array([1, 1])

    def calculate_parameters(self):
        points = np.concatenate([line.points for line in self.data])
        self.x_max, self.y_max = np.amax(points, axis=0)
        self.x_min, self.y_min = np.amin(points, axis=0)

        if self.settings['x_max'] != None: self.x_max = self.settings['x_max']
        if self.settings['x_min'] != None: self.x_min = self.settings['x_min']
        if self.settings['y_max'] != None: self.y_max = self.settings['y_max']
        if self.settings['y_min'] != None: self.y_min = self.settings['y_min']

        dy = self.y_max-self.y_min
        if not dy: dy = 1
        dx = self.x_max-self.x_min
        if not dx: dx = 1

        self.scale = np.array([self.width/dx, self.height/dy])
        self.origin = np.array([-self.x_min*self.scale[0], self.height + self.y_min*self.scale[1]])

    def plot(self, x_data, y_data, style='-', color=(0,0,0), w=2):
        self.data.append(LineData(np.stack((x_data, y_data), axis=1), style, color, w))
        self.calculate_parameters()

        self.surface.fill(self.bg_color)
        for line in self.data:
        	points = (line.points*self.scale*self.horientation + self.origin).astype(int)
        	if '-' in line.style and len(points)>=2: pg.draw.lines(self.surface, line.color, False, points, line.width)
        	if 'o' in line.style:
        		for x, y in points: pg.draw.ellipse(self.surface, line.color,
        			[int(x-line.width), int(y-line.width), line.width*2, line.width*2])
        	if line.style == 'hist':
        		w = self.width//len(points)
        		for x, y in points:
        			pg.draw.polygon(self.surface, line.color, [(x-w//2, y), (x+w//2, y), (x+w//2, self.height), (x-w//2, self.height)])
        if self.grid: self.draw_grid()

    def show(self, root):
        if self.xymouse and pg.Rect(self.position, self.xymouse_surface.get_size()).collidepoint(pg.mouse.get_pos()):
            self.xymouse_surface.fill(pg.SRCALPHA)
            pg.mouse.set_visible(False)
            mx, my = mouse_pos = np.array(pg.mouse.get_pos()) - np.array(self.position)
            pg.draw.line(self.xymouse_surface, (255,100,100), (mx-5, my), (mx+5, my))
            pg.draw.line(self.xymouse_surface, (255,100,100), (mx, my-5), (mx, my+5))
            x, y = (mouse_pos-self.origin)/self.scale*self.horientation
            text = self.axis_font.render(f'x: {x:.2f}   y: {y:.2f}  ', False, self.xymouse_color)
            self.xymouse_surface.blit(text, (
                mx-text.get_width() if mx-text.get_width()>0 else mx+10,
                my-text.get_height() if my-text.get_height()>0 else my+10
            ))
        else:
            self.xymouse_surface.fill(pg.SRCALPHA)
            pg.mouse.set_visible(True)
        root.blit(self.surface, self.position)
        root.blit(self.xymouse_surface, self.position)

    def clear(self):
        self.data = []
        self.surface.fill(self.bg_color)

    def set_grid(self, value): self.grid = value
    def set_xycursor(self, value): self.xymouse = value

    def set_y_min(self, min): self.settings['y_min'] = min

    def draw_grid(self, w=10):
        for i in nrange(w):
            x = self.width//w * i
            y = self.height//w * i
            pg.draw.line(self.surface, self.axis_color, (x, 0), (x, self.height))
            self.surface.blit(self.axis_font.render(f'{((x-self.origin[0])/self.scale[0])*self.horientation[0]:.2f}', False, self.axis_color), (x+1, self.height-w))
            pg.draw.line(self.surface, self.axis_color, (0, y), (self.width, y))
            self.surface.blit(self.axis_font.render(f'{((y-self.origin[1])/self.scale[1])*self.horientation[1]:.2f}', False, self.axis_color), (w, y+1))

class Label: # facilitar adição de texo no pygame
    def __init__(self, text, position, font, color=(0,0,0)):
        self.t = text
        self.text = font.render(text, False, color)
        self.rect = self.text.get_rect()
        self.rect.topleft = position
        self.font = font

    def draw(self, surface): surface.blit(self.text, self.rect)

    def set_text(self, text):
        self.text = self.font.render(text, False, (0,0,0))
