import math

import pygame
import random

colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]


class Figure:
    x = 0
    y = 0

    # figures with all their rotations
    figures = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])


class Tetris:
    level = 2
    score = 0
    full_lines = 0
    state = "start"
    field = []
    height = 0
    width = 0
    x = 250
    y = 0
    zoom = 20
    figure = None
    next_figure = None

    def __init__(self, height, width):
        self.height = height
        self.width = width
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(0)
            self.field.append(new_line)

    def new_figure(self):
        if self.next_figure != None:
            self.figure = self.next_figure
        else:
            self.figure = Figure(3, 0)
        self.next_figure = Figure(3, 0)

    # checks if figure is out of bounds or intesects non empty field
    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                self.full_lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += lines ** 2

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines()
        self.new_figure()
        if self.intersects():
            game.state = "gameover"

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation


# Initialize the game engine
pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
ORANGE = (255, 165, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

size = (600, 500)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Tetris")

# Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()
fps = 25
game = Tetris(20, 10)
counter = 0

pressing_down = False

while not done:
    if game.figure is None:
        game.new_figure()
    counter += 1
    if counter > 100000:
        counter = 0

    if counter % (fps // game.level // 2) == 0 or pressing_down:
        if game.state == "start":
            game.go_down()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                game.rotate()
            if event.key == pygame.K_DOWN:
                pressing_down = True
            if event.key == pygame.K_LEFT:
                game.go_side(-1)
            if event.key == pygame.K_RIGHT:
                game.go_side(1)
            if event.key == pygame.K_SPACE:
                game.go_space()
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                pressing_down = False

    screen.fill(BLACK)

    for i in range(0,game.height):
        for j in range(0,game.width):
            pygame.draw.circle(screen, BLUE, [game.x + game.zoom * j, game.y + game.zoom * i], 1)
            if game.field[i][j] > 0:
                pygame.draw.rect(screen, colors[game.field[i][j]],
                                 [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])
            # draw edges
            if (i == 0 or j == 0) and (i != game.height-1 or j != game.width-1):
                pointlist_3 = [( game.x + game.zoom * j+5,game.y + game.zoom * i), (game.x + game.zoom * j, game.y + game.zoom * i+5), ( game.x + game.zoom * j, game.y + game.zoom * i-5)]
                pygame.draw.polygon(screen, BLUE, pointlist_3, 0)
            # # manualy add last edges
            # pointlist_3 = [(game.x + game.zoom * 0 + 5, game.y + game.zoom * game.height),
            #                (game.x + game.zoom * 0, game.y + game.zoom * game.height + 5),
            #                (game.x + game.zoom * 0, game.y + game.zoom * game.height - 5)]
            # pygame.draw.polygon(screen, BLUE, pointlist_3, 0)
            # pointlist_3 = [(game.x + game.zoom * game.width + 5, game.y + game.zoom * 0),
            #                (game.x + game.zoom * game.width, game.y + game.zoom * 0 + 5),
            #                (game.x + game.zoom * game.width, game.y + game.zoom * 0 - 5)]
            # pygame.draw.polygon(screen, BLUE, pointlist_3, 0)
            # pygame.draw.polygon(screen, BLUE, pointlist_3, 0)


    if game.figure is not None:
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.figure.image():
                    pygame.draw.rect(screen, colors[game.figure.color],
                                     [game.x + game.zoom * (j + game.figure.x) + 1,
                                      game.y + game.zoom * (i + game.figure.y) + 1,
                                      game.zoom - 2, game.zoom - 2])
    if game.next_figure is not None:
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.next_figure.image():
                    pygame.draw.rect(screen, colors[game.next_figure.color],
                                     [game.zoom * (j + game.next_figure.x),
                                      game.zoom * (i + game.next_figure.y) + 350,
                                      game.zoom - 2, game.zoom - 2])

    # text
    help_font = pygame.font.SysFont('Lucida Console', 18, True, False)
    font = pygame.font.SysFont('Lucida Console', 18, True, False)
    font1 = pygame.font.SysFont('Lucida Console', 65, True, False)

    # score text

    text_level = font.render("Your level: " + str(game.level), True, WHITE)
    text_lines = font.render("Full lines: " + str(game.full_lines), True, WHITE)
    text_score = font.render("  Score ", True, WHITE)
    score = font.render(str(game.score), True, ORANGE)
    text_game_over = font1.render("Game Over :( ", True, (255, 0, 0))
    screen.blit(text_level, [0, 10])
    screen.blit(text_lines, [0, 30])
    screen.blit(text_score, [0, 60])
    screen.blit(score, [145, 60])

    # help text
    text_help = font.render("H E L P", True, WHITE)
    text_help_left = help_font.render("Swipe Left: LEFT", True, WHITE)
    text_help_right = help_font.render("Swipe Right: RIGHT", True, WHITE)
    text_help_down = help_font.render("Swipe Down: DROP", True, WHITE)
    text_help_rotate = help_font.render("Swipe Up: ROTATE", True, WHITE)
    text_help_swap = help_font.render("Thumbs Down: SWAP", True, WHITE)
    play_tetris = help_font.render("Play TETRIS !", True, RED)

    next = help_font.render("Next : ", True, WHITE)

    screen.blit(text_help, [70, 130])
    screen.blit(text_help_left, [10, 170])
    screen.blit(text_help_right, [10, 190])
    screen.blit(text_help_down, [10, 210])
    screen.blit(text_help_rotate, [10, 230])
    screen.blit(text_help_swap, [10, 250])
    screen.blit(play_tetris, [280, 440])

    screen.blit(next, [70, 320])

    if game.state == "gameover":
        screen.blit(text_game_over, [10, 200])

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
