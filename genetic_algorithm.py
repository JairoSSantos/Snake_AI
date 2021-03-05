from game_classes import *
from neural_network import *

def new_training(game_kwargs, plt, size_population, eta, description, keep_parents, keep_best):
    filename = 'training_0' # iniciar nome do arquivo default
    while filename + '.pkl' in os.listdir(SAVE_TRAINING_FOLDER): # enquanto houver um arquivo igual
        filename, u = filename.split('_')
        filename = '_'.join([filename, str(int(u)+1)]) # modifique o nome do arquivo

    nns_filename = SAVE_NN_FOLDER + filename + '/'
    training_settings = {
        'nns_path': nns_filename, # caminho para a pasta de salvamento do treinamento
        'evolution_path': SAVE_EVOLUTION_FOLDER + filename + '.csv', # caminho para o arquivo csv dos dados de evolução
        'eta': eta, # taxa de mutação
        'size_population': size_population, # tamanho da população
        'lifes': game_kwargs['lifes'], # chances de cada indivíduo
        'description':description, # descrição do treinamento
        'keep_parents': keep_parents,
        'keep_best': keep_best
    }
    pd.DataFrame().to_csv(training_settings['evolution_path'], index=False) # salvar arquivo da evolução

    if not filename in os.listdir(SAVE_NN_FOLDER): os.mkdir(nns_filename) # verificar se ja existe uma pasta para o treinamento
    elif len(os.listdir(nns_filename)):  # se a pasta conter arquivos então remova-os
        for item in os.listdir(nns_filename): os.remove(os.path.join(nns_filename, item))

    filename = SAVE_TRAINING_FOLDER + filename + '.pkl' # caminho para o arquivo de treinamento
    with open(filename, 'wb') as file: pkl.dump(training_settings, file) # salvar treinamento

    return Environment(game_kwargs, plt, size_population, eta, training_settings['nns_path'],
        training_settings['evolution_path'], description, filename, keep_parents, keep_best)

def load_training(filename, game_size, plt):
    with open(os.path.join(SAVE_TRAINING_FOLDER, filename), 'rb') as file: training_settings = pkl.load(file)

    training = Environment({'width': game_size[0], 'height': game_size[1], 'lifes':training_settings['lifes']},
        plt, training_settings['size_population'],
        training_settings['eta'],
        training_settings['nns_path'],
        training_settings['evolution_path'],
        training_settings['description'],
        os.path.join(SAVE_TRAINING_FOLDER, filename),
        training_settings['keep_parents'],
        training_settings['keep_best'])

    training.load_networks() # carregar redes salvas
    training.evolution = pd.read_csv(training_settings['evolution_path']).to_dict('list') # pegar dados da evolução
    training.generation = len(training.evolution['medians'])
    training.plt.set_y_min(0)
    training.plt.set_grid(True)
    training.info_update()
    training.data_plot()

    return training

class Environment:
    def __init__(self, game_kwargs, plt, size_population, eta, nns_path, evolution_path, description, path, keep_parents, keep_best):
        self.game_kwargs = game_kwargs # parâmetros da classe jogo
        self.plt = plt # classe para plotar grafico 'class Graphic - settings.py'
        self.size_population = size_population # tamanho de cada população
        self.eta = eta # taxa de mutação
        self.evolution = {'medians':[], 'bests_scores':[]} # armazenar informações de treinamento
        self.generation = 0 # numero de gerações
        self.n_parents = 2 # numero de indivíduos que serão cruzados
        self.population = self.individuals([NeuralNetwork() for _ in nrange(self.size_population)], False) # população

        self.evolution_path  = evolution_path # arquivo para salvar dados de evolução
        self.nns_path = nns_path # pasta salvar redes neurais
        self.path = path

        self.is_running = True # se falso o trinamento é pausado

        self.keep_parents = keep_parents
        self.keep_best = keep_best

        self.description = description
        self.info = [self.generation, 0, 0, 0, 0, self.eta] # informações [geração atual, melhor pontuação, média da ultima geração, melhor pontuação atual, ainda jogando]

    def individuals(self, networks, cross): # gerar uma nova população a partir do cruzamento dos genitores
        return [
            {'game':SnakeGame(**self.game_kwargs),
             'network': self.crossover(networks)}
            for _ in nrange(self.size_population)
        ] if cross else [
            {'game':SnakeGame(**self.game_kwargs),
             'network': network}
            for network in networks
        ]

    def crossover(self, networks): # cruzamento genético
        nn = NeuralNetwork()
        cross_point = len(nn.weights)//len(networks) # selecionar ponto de cruzamento
        shuffle(networks) # embaralhar as redes
        nn1, nn2 = networks
        nn.weights = nn1.weights[:cross_point] + nn2.weights[cross_point:]
        nn.biases = nn2.biases[:cross_point] + nn1.biases[cross_point:]
        nn.mutate(self.eta) # adicionar multação a rede filho
        return nn

    def draw(self, surface, nn_surface):
        population_scores = [item['game'].get_score() if item['game'].is_running else 0 for item in self.population] # pontuações da população
        best_score = sorted(population_scores)[-1] # melhor pontuação
        self.info[3] = f'{best_score:.2f}'
        index = population_scores.index(best_score)
        self.population[index]['game'].draw(surface) # desenhar jogo com melhor pontuação
        self.population[index]['network'].draw(nn_surface)

    def info_update(self): # atualizar informações sobre treinamento
        x1, x2, x3 = (
            sorted(self.evolution['bests_scores'])[-1],
            self.evolution['bests_scores'].index(sorted(self.evolution['bests_scores'])[-1]),
            self.evolution['medians'][-1]
        )
        self.info = [self.generation, # geração
            f'{x1:.2f} - {x2}', # melhor pontuação
            f'{x3:.2f}', # media da ultima geração
            0, 0, self.eta]

    def load_networks(self): # carregar redes salvas
        networks = []
        for i in nrange(self.n_parents):
            try: networks.append(import_network(os.path.join(self.nns_path, f'parent-{i}.pkl')))
            except: pass
        self.population = self.individuals(networks, True) # cruzamento dos indivíduos
        if self.keep_parents: self.population += self.individuals(networks, False) # manter os indivíduos
        if self.keep_best: self.population += self.individuals([self.load_best_network()], False) # melhor indivíduo alcançado

    def load_best_network(self):
        nns = os.listdir(self.nns_path)
        nns.remove('parent-0.pkl')
        nns.remove('parent-1.pkl')
        bid = 0; nn_f = None
        for nn in nns:
            if int(nn.rstrip('.pkl').split('-')[-1]) > bid: nn_f = nn; bid = int(nn.rstrip('.pkl').split('-')[-1])
        return import_network(os.path.join(self.nns_path, nn_f))

    def new_population(self): # gerar nova população
        population_scores = [item['game'].get_score() for item in self.population] # coletar pontuação dos indivíduos
        networks = [self.population[population_scores.index(sorted(population_scores)[-1-i])]['network'] for i in nrange(self.n_parents)] # selecionar os melhores indivíduos

        self.evolution['medians'].append(np.median(population_scores)) # armazenar média das populações
        best_score = sorted(population_scores)[-1] # melhor pnotuação
        if not len(self.evolution['bests_scores']) or (len(self.evolution['bests_scores']) and best_score > sorted(self.evolution['bests_scores'])[-1]):
            networks[0].save(os.path.join(self.nns_path, f'best-{self.generation}.pkl')) # salvar melhor rede da geração
            print('saved')
        self.evolution['bests_scores'].append(best_score) # armazenar melhor pontuação
        for i in nrange(self.n_parents): networks[i].save(os.path.join(self.nns_path, f'parent-{i}.pkl')) # salvar redes genitoras

        self.population = self.individuals(networks, True) # cruzamento dos indivíduos
        if self.keep_parents: self.population += self.individuals(networks, False) # manter os indivíduos
        if self.keep_best: self.population += self.individuals([self.load_best_network()], False) # melhor indivíduo alcançado

        pd.DataFrame(self.evolution).to_csv(self.evolution_path, index=False) # salvar dados da evolução

        self.data_plot() # mostrar dados
        self.info_update() # atualizar informações sobre o treinamento
        self.generation += 1

        print(f'Nova geração --> {self.generation}')

    def play(self): self.is_running = True

    def data_plot(self): # plotar gráfico
        self.plt.clear() # limpar figura
        x_ = range(len(self.evolution['medians'])) # dados do eixo x
        self.plt.plot(x_, self.evolution['medians'], style='hist', color=PLT_COLORS[0]) # plotar média
        self.plt.plot(x_, self.evolution['bests_scores'], style='o', color=PLT_COLORS[1]) # plotar melhor pontuação
        self.plt.plot(x_, moving_average(self.evolution['medians'], 20), color=PLT_COLORS[2]) # plotar média móvel das médias
        self.plt.plot(x_, moving_average(self.evolution['bests_scores'], 20), color=PLT_COLORS[3]) # plotar média móvel das melhores pontuações

    def save(self):
        training_settings = {
            'nns_path': self.nns_path, # caminho para a pasta de salvamento do treinamento
            'evolution_path': self.evolution_path, # caminho para o arquivo csv dos dados de evolução
            'eta': self.eta, # taxa de mutação
            'size_population': self.size_population, # tamanho da população
            'lifes': self.game_kwargs['lifes'], # chances de cada indivíduo
            'description': self.description, # descrição do treinamento
            'keep_parents': self.keep_parents,
            'keep_best': self.keep_best
        }
        with open(self.path, 'wb') as file: pkl.dump(training_settings, file)

    def stop(self): self.is_running = False

    def update(self):
        if self.is_running:
            games_running = 0 # contar quantos jogos ainda não perderam
            for item in self.population:
                game, network = item.values()
                game.update() # atualizar jogo
                if game.is_running:
                    games_running += 1
                    game.feedforward(network) # propagar sinal e executar ação
            self.info[4] = f'{games_running}/{self.size_population}'
            if not games_running: self.new_population() # se não houver jogos ainda rodando atualize a população
