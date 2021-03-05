from settings import *

MAX_VALUE = 999999
RANGE_INITIAL_VALUES = -1, 1

RELU = lambda x: np.clip(x, 0, MAX_VALUE) # relu
SIG = lambda x: 1 / (1 + np.exp(-x)) # sigmoid
SOFT = lambda x: np.exp(x)/sum(np.exp(x)) # softmax
POT = lambda x: x/sum(x)

def import_network(path):
    network = NeuralNetwork()
    try:
        with open(path, 'rb') as file: network.weights, network.biases = list(pkl.load(file).values())
        print(f'Rede {path} carregada')
    except: print('Erro ao carregar rede neural')
    return network

class NeuralNetwork:
    def __init__(self, layout=NN_LAYOUT):
        self.layout = layout
        self.weights = [] # armazenar pesos
        self.biases = [] # armazenar vieses
        self.activations_functions = [] # armazenar as funções de ativação para cada camada
        self.outputs = [] # armazenar as saídas da rede
        arange = np.linspace(*RANGE_INITIAL_VALUES, 1000) # variação dos valores iniciais para a rede
        for n_neurons, n_inputs in zip(layout[1:], layout):
            self.weights.append(np.random.choice(arange, (n_neurons, n_inputs))) # matrix de pesos
            self.biases.append(np.random.choice(arange, n_neurons)) # vetor de vieses
            self.activations_functions.append(RELU) # função de ativação da camada
        # self.activations_functions[-1] = POT # função de ativação da ultima camada

    def draw(self, surface):
        width = surface.get_width()
        height = surface.get_height()
        dW = width/len(self.layout) # dividir a largura da superfície pela quantidade de camadas
        mc = np.amax(self.layout)
        dH = [height/nc for nc in self.layout] # dividir a altura da superfície pelo numero de neurônios para cada camada
        r = np.amin(dH + [dW])# o diâmetro dos neurônios será a menor parte encontrada
        ellipses = [] # armazenar rects e cores dos neurônios
        lines = [] # armazenar pontos para desenhar os pesos
        for x, nc in enumerate(self.layout):
            lay = [] # armazenar os pontos das útimas camadas
            try: outs_min, outs_max = np.amin(self.outputs[x]), np.amax(self.outputs[x]) # tente encontrar os limites de cada saída
            except IndexError: outs_min, outs_max = 0, 1 # se não conseguir definir como 0 e 1
            for y in nrange(nc):
                try: color = int(rmap(self.outputs[x][y], outs_min, outs_max, 0, 255)) # mapear a cor de acordo com o valor de saída
                except: color = 255 # se der algum erro defina como 255
                rect = pg.Rect(int(x*dW), int(y*dH[x]), int(r), int(r)) # rect do neurônio
                rect.center = pg.Rect(int(x*dW), int(y*r + (mc-nc)//2*r), int(dW), int(r)).center # posicionar rect
                lay.append(rect.center)
                for p in lines:pg.draw.line(surface, (120,120,120), rect.center, p) # desenhar pesos
                ellipses.append((rect, (color, color, 50)))
            lines = lay[:]
        for rect, color in ellipses: pg.draw.ellipse(surface, color, rect) # desenhar neurônios

    def forward(self, a): # propagação do sinal
        self.outputs = [a]
        for w, b, f in zip(self.weights, self.biases, self.activations_functions):
            a = f(np.dot(w, a) + b)
            self.outputs.append(a)
        return a

    def mutate(self, eta):
        arange = np.linspace(-eta, eta, 1000) # variação da mutação
        for i, w, b in zip(range(len(self.weights)), self.weights, self.biases):
            self.weights[i] = w + np.random.choice(arange, w.shape)*np.random.choice([0, 1], w.shape)
            self.biases[i] = b + np.random.choice(arange, b.shape)*np.random.choice([0, 1], b.shape)

    def save(self, path):
        with open(path, 'wb') as file:
            pkl.dump(
                {'weights': self.weights,
                 'biases': self.biases},
                file
            )
