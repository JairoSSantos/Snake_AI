from settings import *

class Snake:
    def __init__(self, x, y, initial_lenght, initial_direction, color=(0,0,0)):
        self.lenght = initial_lenght
        self.direction = initial_direction
        self.color = color
        self.motion = {'u':pg.math.Vector2(0, -1), 'd':pg.math.Vector2(0, 1),
                       'r':pg.math.Vector2(1, 0), 'l':pg.math.Vector2(-1, 0)}
        self.body = [pg.math.Vector2(x, y)]

    def draw(self, surface):
        hmax = len(self.body)
        for h, pos in enumerate(self.body):
            c = rmap(h, 0, hmax, 0, 255)
            pg.draw.rect(surface, (c, 255, 255-c), [pos*SCALE, (SCALE, SCALE)])

    def get_head(self): return self.body[-1]

    def update(self):
        self.body.append(self.body[-1] + self.motion[self.direction])
        if len(self.body) > self.lenght: self.body.pop(0)

class SnakeGame:
    def __init__(self, width, height, lifes, initial_lenght=5, initial_direction='r'):
        self.size = self.width, self.height = width-1, height-1 # tamanho do jogo em escala reduzida -> (tamanho real pixels / escala)
        self.initial_lenght = initial_lenght # comprimento inicial da cobrinha
        self.initial_direction = initial_direction # direção inicial
        self.snake_color = SNAKE_COLOR # cor da cobrinha
        self.apple_color = APPLE_COLOR # cor da maçã

        self.snake = None
        self.apple = None
        self.new_snake() # nova snake
        self.new_apple() # nova maçã

        self.score = 0 # pontuação
        self.lifes = lifes # vida da cobrinha
        self.lifes0 = lifes
        self.apple_score = APPLE_SCORE # quantidade de pontuação a ser adicionada quando a cobrinha pegar maçã
        self.distance_to_apple = self.snake.get_head().distance_to(self.apple) # distância entre a cobrinha e maçã
        self.step_cont = 0 # contagem de passos
        self.step_lim = STEP_LIM # limite de passos sem comer a maçã até morrer

        self.is_running = True

    def change_snake_direction(self, new_direction):
        for d, rd in zip(['u', 'd', 'r', 'l'], ['d', 'u', 'l', 'r']):
            if new_direction == d and self.snake.direction != rd: # verificar se a nova direção não é contrária a direção atual
                self.snake.direction = new_direction
                break

    def draw(self, surface):
        pg.draw.ellipse(surface, self.apple_color, [self.apple*SCALE, (SCALE, SCALE)])
        self.snake.draw(surface)

    def feedforward(self, network):
        output = dict(zip(
            network.forward(self.get_sensors()), ['u', 'd', 'r', 'l'])) # ligar cada saída ao respectivo movimento
        self.change_snake_direction(output[sorted(output.keys())[-1]]) # mudar direção de snake de acordo com a maior saída da rede

    def get_score(self): return self.score/self.lifes0

    def get_sensors(self):
        values = []
        x, y = self.snake.get_head()

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i or j:
                    values.append(1 if x+i>self.width or x+i<0 or y+j>self.height or y+j<0 or (x+i, y+j) in self.snake.body[:-1] else 0)
                    '''i, j = i*2, j*2
                    values.append(1 if x+i>self.width or x+i<0 or y+j>self.height or y+j<0 or (x+i, y+j) in self.snake.body[:-1] else 0)'''

        values.append(x > self.apple.x)
        values.append(y > self.apple.y)

        for d in ['u', 'd', 'r', 'l']: values.append(self.snake.direction == d)

        return np.array(values)

    def new_apple(self):
        self.apple = pg.math.Vector2(randint(0, self.width), randint(0, self.height))
        cont = 0
        while self.apple in self.snake.body:
            self.apple = pg.math.Vector2(randint(0, self.width), randint(0, self.height))
            if cont > 1000: break
            cont += 1

    def new_snake(self):
        self.snake = Snake(self.width//2, self.height//2,
            self.initial_lenght, self.initial_direction, self.snake_color)

    def play(self): self.is_running = True

    def restart(self): self.play(); self.score = 0; self.lifes = self.lifes0; self.step_cont = 0

    def stop(self): self.is_running = False

    def update(self):
        if self.is_running:
            self.snake.update() # atualizar posição snake
            self.step_cont += 1 # contar passos
            # self.score += 0.1 if self.step_cont < self.step_lim/2 else -0.1 # pontuar movimento

            if self.snake.get_head() == self.apple: # verificar colisão com apple
                self.score += self.apple_score
                self.snake.lenght += 1
                self.step_cont = 0  # resetar contagem de passos
                self.new_apple()

            distance_to_apple = self.snake.get_head().distance_to(self.apple)
            if distance_to_apple < self.distance_to_apple:
                self.score += 1 # pontuar aproximação com maçã
                self.distance_to_apple = distance_to_apple

            x, y = self.snake.get_head()
            it_lost = ( # condições para perder
                x > self.width or x < 0 or y > self.height or y < 0,  # colisão com bordas
                self.snake.get_head() in self.snake.body[:-1], # colisão com sigo mesmo
                self.step_cont > self.step_lim # passou do limite de passos
            )
            if any(it_lost):
                self.lifes -= 1
                self.new_snake()
                self.new_apple()
                if self.lifes <= 0: self.is_running = False
