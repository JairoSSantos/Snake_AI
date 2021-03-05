import tkinter as tk
from tkinter import ttk
from genetic_algorithm import *

GAME_SIZE = GAME_WIDTH, GAME_HEIGHT = 600, 600
PAD = 20
PLT_POS = GAME_WIDTH+PAD, PAD//2
PLT_SIZE = 650, 350

fonts = [
    ('calibre', 14),
    ('calibre', 12)
]

def init():
    root = tk.Tk()
    App(root)
    root.title('Snake AI')
    root.mainloop()

def modify(training):
    root =  tk.Tk()
    AppModify(root, training)
    root.title('Modificar')
    root.mainloop()

def play(network=None):
    player = False if network else True
    game_surf = pg.Surface(GAME_SIZE)
    pg.display.set_caption(f'Snake Game') # definir título
    screen = pg.display.set_mode((GAME_WIDTH if player else int(GAME_WIDTH*5/3), GAME_HEIGHT)) # surface principal

    game = SnakeGame(GAME_WIDTH//SCALE, GAME_HEIGHT//SCALE, 1)
    if player: game.stop()
    else:
        label_pontuacao = Label('Pontuação: __________', (GAME_WIDTH+PAD, PAD), pg.font.SysFont('courier', 24, bold=True))
        nn_surf = pg.Surface((300,350))
    timer = pg.time.Clock()

    run = True
    while run:
        if player:
            timer.tick(10)
            pg.display.set_caption(f'Pontuação: {game.get_score()}')

        for event in pg.event.get():
            if event.type == pg.QUIT: quit()
            elif event.type == pg.KEYDOWN:
                if player:
                    try:game.change_snake_direction({pg.K_UP:'u', pg.K_DOWN:'d', pg.K_RIGHT:'r', pg.K_LEFT:'l'}[event.key])
                    except KeyError:
                        if event.key == pg.K_ESCAPE: run = False
                        elif event.key == pg.K_SPACE: game.restart()
                else:
                    if event.key == pg.K_ESCAPE: run = False
                    elif event.key == pg.K_SPACE: game.restart()

        screen.fill(BACKGROUND_TRAINING_COLOR)
        game_surf.fill(BACKGROUND_GAME_COLOR)

        game.update()
        game.draw(game_surf)

        if not player:
            if game.is_running: game.feedforward(network)
            else: game.restart()
            nn_surf.fill(BACKGROUND_TRAINING_COLOR)
            network.draw(nn_surf)
            label_pontuacao.set_text(label_pontuacao.t.rstrip('__________') + '{0:.2f}'.format(game.get_score()))
            label_pontuacao.draw(screen)
            screen.blit(nn_surf, (GAME_WIDTH+PAD, label_pontuacao.rect.bottom + PAD))

        screen.blit(game_surf, (0,0))

        # pg.image.save(screen, os.path.join('imgs', str(len(os.listdir('imgs/'))) + '.png' ))

        pg.display.flip()
    pg.display.quit()
    init()

def training_play(training):
    game = pg.Surface(GAME_SIZE)
    nn_surface = pg.Surface((300,350))

    plt = training.plt
    plt.set_y_min(0)
    plt.set_grid(True)

    title = training.path.split('/')[-1]
    pg.display.set_caption(f'Treinamento - {title}') # definir título
    screen = pg.display.set_mode((0,0), pg.RESIZABLE) # surface principal

    font1_size = 16  # tamanho da font1
    text_pad = 5
    font1 = pg.font.SysFont('courier', font1_size)
    labels = [
        Label(label, (GAME_WIDTH+PAD+nn_surface.get_width()+text_pad, plt.height + PAD + (font1_size + text_pad)*i + text_pad), font1)
        for i, label in enumerate([
            'Geração:__________',
            'Melhor pontuação:__________',
            'Média da última geração:__________',
            'Pontuação:__________',
            'Jogando:__________',
            'Taxa de mutação:__________'
        ])
    ]
    rect_labels = pg.Rect(labels[0].rect.topleft, (1, 1)).unionall([label.rect for label in labels]).inflate(text_pad*2, text_pad*2)
    k_label = Label('Espaço: pause/play; Esc: menu; F6: modificar',
        (PAD//2, GAME_HEIGHT+PAD), pg.font.SysFont('courier', 14, italic=True))\

    run = True
    show = True
    arrays = []
    while run:
        # pg.time.Clock().tick(10)
        for event in pg.event.get():
            if event.type == pg.QUIT: quit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    if training.is_running: training.stop(); plt.set_xycursor(True)
                    else: training.play(); plt.set_xycursor(False)
                elif event.key == pg.K_ESCAPE: run = False
                elif event.key == pg.K_F6: modify(training)
                elif event.key == pg.K_F8: show = False if show else True

        training.update()

        if show:
            screen.fill(BACKGROUND_TRAINING_COLOR)
            game.fill(BACKGROUND_GAME_COLOR)
            nn_surface.fill(BACKGROUND_TRAINING_COLOR)

            training.draw(game, nn_surface)
            screen.blit(game, (PAD//2, PAD//2))
            screen.blit(nn_surface, (GAME_WIDTH+PAD, plt.height+PAD))

            plt.show(screen)

            pg.draw.rect(screen, (150,150,150), rect_labels, 1)
            for item, text in zip(labels, training.info):
                item.set_text(item.t.strip('_') + str(text))
                item.draw(screen)
                if labels.index(item) < len(labels)-1: pg.draw.line(screen, (150,150,150),
                    (rect_labels.x, item.rect.bottom + text_pad//2), (rect_labels.right, item.rect.bottom + text_pad//2))
            k_label.draw(screen)

            # pg.image.save(screen, os.path.join('imgs', str(len(os.listdir('imgs/'))) + '.png' ))

            pg.display.flip()
    pg.display.quit()
    init()

class AppModify:
    def __init__(self, root, training):
        self.root = root
        self.training = training
        self.items = [
            'Quantidade de invivíduos por geração:',
            'Taxa de mutação:',
            'Chances de cada indivíduo:',
            'Descrição:'
        ]
        for (i, label), value in zip(enumerate(self.items), [training.size_population, training.eta, training.game_kwargs['lifes'], training.description]):
            label_frame = tk.Frame(root)
            label_frame.pack(side=tk.TOP, pady=5, padx=10)
            label = tk.Label(label_frame, text=label, font=fonts[1])
            label.pack(side=tk.LEFT)
            entry = tk.Entry(label_frame, font=fonts[1])
            entry.insert(0, str(value))
            entry.pack(side=tk.RIGHT)
            self.items[i] = {'frame':label_frame, 'label':label, 'entry':entry}
        self.items[0]['entry'].focus_set()

        self.check_frame = tk.Frame(root)
        self.check_frame.pack(side=tk.TOP)
        self.vars = {'Manter os pais da última geração':tk.IntVar(), 'Manter o melhor indivíduo':tk.IntVar()}
        for (text, var), value in zip(self.vars.items(), [training.keep_parents, training.keep_best]):
            var.set(value)
            check = tk.Checkbutton(self.check_frame, text=text, var=var)
            check.pack()

        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack(pady=5)

        self.button_init = tk.Button(self.buttons_frame, text='Salvar', font=fonts[0], command=self.save)
        self.button_init.pack(side=tk.LEFT, padx=5)

        self.button_cancel = tk.Button(self.buttons_frame, text='Calcelar', font=fonts[0], command=self.cancel)
        self.button_cancel.pack(side=tk.LEFT, padx=5)

    def save(self):
        try:
            size_population = int(self.items[0]['entry'].get())
            eta = float(self.items[1]['entry'].get())
            lifes = int(self.items[2]['entry'].get())
            description = str(self.items[3]['entry'].get())
            keep_parents, keep_best = [bool(var.get()) for var in self.vars.values()]
        except Exception as error: tk.messagebox.showerror(title='Erro', message=error)
        else:
            self.root.destroy()
            self.training.size_population = size_population
            self.training.eta = eta
            self.training.game_kwargs['lifes'] = lifes
            self.training.description = description
            self.training.keep_parents = keep_parents
            self.training.keep_best = keep_best
            self.training.save()

    def cancel(self):
        self.root.destroy()

class ToplevelNewtraining:
    def __init__(self, root, root_main):
        self.root = root
        self.root_main = root_main
        self.items = [
            'Quantidade de invivíduos por geração:',
            'Taxa de mutação:',
            'Chances de cada indivíduo:',
            'Descrição:'
        ]
        for (i, label), value in zip(enumerate(self.items), [SIZE_POPULATION, ETA, LIFES, '']):
            label_frame = tk.Frame(root)
            label_frame.pack(side=tk.TOP, pady=5, padx=10)
            label = tk.Label(label_frame, text=label, font=fonts[1])
            label.pack(side=tk.LEFT)
            entry = tk.Entry(label_frame, font=fonts[1])
            entry.insert(0, str(value))
            entry.pack(side=tk.RIGHT)
            self.items[i] = {'frame':label_frame, 'label':label, 'entry':entry}
        self.items[0]['entry'].focus_set()

        self.check_frame = tk.Frame(root)
        self.check_frame.pack(side=tk.TOP)
        self.vars = {'Manter os pais da última geração':tk.IntVar(), 'Manter o melhor indivíduo':tk.IntVar()}
        for text, var in self.vars.items():
            var.set(True)
            check = tk.Checkbutton(self.check_frame, text=text, var=var)
            check.pack()

        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack(pady=5)

        self.button_init = tk.Button(self.buttons_frame, text='Iniciar', font=fonts[0], command=self.init)
        self.button_init.pack(side=tk.LEFT, padx=5)

        self.button_cancel = tk.Button(self.buttons_frame, text='Calcelar', font=fonts[0], command=self.cancel)
        self.button_cancel.pack(side=tk.LEFT, padx=5)

    def init(self):
        try:
            size_population = int(self.items[0]['entry'].get())
            eta = float(self.items[1]['entry'].get())
            lifes = int(self.items[2]['entry'].get())
            description = str(self.items[3]['entry'].get())
            keep_parents, keep_best = [bool(var.get()) for var in self.vars.values()]
        except Exception as error: tk.messagebox.showerror(title='Erro', message=error)
        else:
            self.root.destroy()
            self.root.update()
            self.root_main.destroy()
            training_play(new_training(
                {'width':GAME_WIDTH//SCALE, 'height':GAME_HEIGHT//SCALE, 'lifes':lifes},
                Graph(PLT_POS, PLT_SIZE),
                size_population,
                eta,
                description,
                keep_parents,
                keep_best
            ))

    def cancel(self):
        self.root.destroy()
        self.root.update()

class ToplevelTest:
    def __init__(self, root, root_main, filename):
        self.root = root
        self.root_main = root_main

        with open(os.path.join(SAVE_TRAINING_FOLDER, filename), 'rb') as file: training_settings = pkl.load(file)
        evolution = pd.read_csv(training_settings['evolution_path']).to_dict('list')
        self.save_folder = training_settings['nns_path']

        self.nns_tree = ttk.Treeview(root, columns=['rede', 'geração', 'pontuação'], height=20)
        self.nns_tree.heading('rede', text='Rede Neural')
        self.nns_tree.column('rede', width=100)
        self.nns_tree.heading('geração', text='Geração')
        self.nns_tree.column('geração', width=100)
        self.nns_tree.heading('pontuação', text='Pontuação')
        self.nns_tree.column('pontuação', width=100)
        self.nns_tree.column('#0', width=40)
        self.nns_tree.pack()

        for nn_name in os.listdir(training_settings['nns_path']):
            if 'best' in nn_name:
                index = int(nn_name.rstrip('.pkl').split('-')[-1])-1
                self.nns_tree.insert('', 'end', values=[nn_name, index, evolution['bests_scores'][index]])
            else:
                self.nns_tree.insert('', 'end', values=[nn_name, '', ''])

        self.button_toplevel_frame = tk.Frame(root)
        self.button_toplevel_frame.pack()
        self.test_button = tk.Button(self.button_toplevel_frame, text='Testar', font=fonts[0], command=self.init)
        self.test_button.pack(side=tk.LEFT)
        self.cancel_button = tk.Button(self.button_toplevel_frame, text='Cancelar', font=fonts[0], command=self.cancel)
        self.cancel_button.pack(side=tk.LEFT)

    def init(self):
        item = self.nns_tree.focus()
        try: filename = self.nns_tree.item(item)['values'][0]
        except IndexError: pass
        else:
            self.root.destroy()
            self.root.update()
            self.root_main.destroy()
            print(os.path.join(self.save_folder, filename))
            play(import_network(os.path.join(self.save_folder, filename)))

    def cancel(self):
        self.root.destroy()
        self.root.update()

class App:
    def __init__(self, root):
        self.root = root
        self.tree_frame = tk.Frame(root)
        self.tree_frame.pack()

        treesettings = {
            'Treinamento': 110,
            'Geração': 60,
            'Melhor pontuação': 125,
            'população': 70,
            'mutação': 60,
            'chances': 60,
            'Descrição':200
        }

        self.tree = ttk.Treeview(self.tree_frame, columns=list(treesettings.keys()), height=20)
        for text, width in treesettings.items():
            self.tree.heading(text, text=text)
            self.tree.column(text, width=width)
        self.tree.column('#0', width=40)
        self.tree.pack()

        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack()

        self.play_button = tk.Button(self.buttons_frame, text='Jogar', font=fonts[0], command=self.play)
        self.play_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.new_training_button = tk.Button(self.buttons_frame, text='Novo Treinamento', font=fonts[0], command=self.toplevel_new_training)
        self.new_training_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.continue_training_button = tk.Button(self.buttons_frame, text='Continuar Treinamento', font=fonts[0], command=self.continue_training)
        self.continue_training_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.test_training_button = tk.Button(self.buttons_frame, text='Testar', font=fonts[0], command=self.test_training)
        self.test_training_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.delete_training_button = tk.Button(self.buttons_frame, text='Excluir', font=fonts[0], command=self.delete_training)
        self.delete_training_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.tree_update()

    def continue_training(self):
        item = self.tree.focus()
        try: filename = self.tree.item(item)['values'][0]
        except IndexError: pass
        else:
            self.root.destroy()
            training_play(load_training(filename, (GAME_WIDTH//SCALE, GAME_HEIGHT//SCALE), Graph(PLT_POS, PLT_SIZE)))

    def delete_training(self):
        item = self.tree.focus()
        try: filename = self.tree.item(item)['values'][0]
        except IndexError: pass
        else:
            self.tree.delete(item)
            os.remove(os.path.join(SAVE_TRAINING_FOLDER, filename))
            os.remove(os.path.join(SAVE_EVOLUTION_FOLDER, filename[:-4]+'.csv'))
            rmtree(os.path.join(SAVE_NN_FOLDER, filename[:-4]))

    def play(self):
        self.root.destroy()
        play()

    def test_training(self):
        item = self.tree.focus()
        try: filename = self.tree.item(item)['values'][0]
        except IndexError: pass
        else:
            toplevel = tk.Toplevel()
            toplevel.title('Testar Rede Neural')
            ToplevelTest(toplevel, self.root, filename)

    def toplevel_new_training(self):
        toplevel = tk.Toplevel()
        toplevel.title('Novo Treinamento')
        ToplevelNewtraining(toplevel, self.root)
        toplevel.mainloop()

    def tree_update(self):
        for i, item in enumerate(os.listdir(SAVE_TRAINING_FOLDER)):
            with open(os.path.join(SAVE_TRAINING_FOLDER, item), 'rb') as file: training_settings = pkl.load(file)
            evolution = pd.read_csv(training_settings['evolution_path']).to_dict('list')
            generation = len(evolution['medians'])
            best = np.amax(evolution['bests_scores'])
            self.tree.insert('', 'end', str(i), text=i, values=[
                item,
                generation,
                best,
                training_settings['size_population'],
                training_settings['eta'],
                training_settings['lifes'],
                training_settings['description']
            ])
        self.tree.pack()

if __name__ == '__main__': init()
