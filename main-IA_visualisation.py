import pygame
import random
import sys
import numpy as np
import io

# Visualisation
import matplotlib
matplotlib.use("Agg")  # backend compatible pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Initialisation ---
pygame.init()
GAME_WIDTH, GAME_HEIGHT = 800, 400   # zone de jeu
GRAPH_WIDTH = 400                    # zone de graphe à droite
WIDTH, HEIGHT = GAME_WIDTH + GRAPH_WIDTH, GAME_HEIGHT
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino AI - Algorithme Génétique (v2 corrigée)")

WHITE = (247, 247, 247)
BLACK = (50, 50, 50)
GROUND_Y = GAME_HEIGHT - 40
FONT = pygame.font.SysFont("Courier New", 20)

# Stats
score_history = []

# --- Réseau de neurones simple avec biais ---
class NeuralNet:
    def __init__(self):
        # 4 -> 6 -> 1 (4 entrées, 6 neurones cachés, 1 sortie)
        self.w1 = np.random.randn(4, 6) # poids couche 1
        self.b1 = np.random.randn(6) # biais couche 1
        self.w2 = np.random.randn(6, 1) # poids couche 2
        self.b2 = np.random.randn(1) # biais couche 2

    def predict(self, inputs):
        hidden = np.tanh(np.dot(inputs, self.w1))  # couche cachée
        output = 1 / (1 + np.exp(-np.dot(hidden, self.w2)))  # sigmoid
        return output[0]

    def crossover(self, partner):
        child = NeuralNet()
        child.w1 = np.where(np.random.rand(*self.w1.shape) < 0.5, self.w1, partner.w1)
        child.b1 = np.where(np.random.rand(*self.b1.shape) < 0.5, self.b1, partner.b1)
        child.w2 = np.where(np.random.rand(*self.w2.shape) < 0.5, self.w2, partner.w2)
        child.b2 = np.where(np.random.rand(*self.b2.shape) < 0.5, self.b2, partner.b2)
        return child

    def mutate(self, rate=0.1, scale=0.5):
        for arr in [self.w1, self.b1, self.w2, self.b2]:
            mask = np.random.rand(*arr.shape) < rate
            arr += mask * np.random.randn(*arr.shape) * scale

# --- Classe Dinosaur ---
class Dinosaur:
    def __init__(self, brain=None, color=None):
        self.size = 50
        self.x = 60
        self.y = GROUND_Y - self.size
        self.velocityY = 0
        self.gravity = 0.7
        self.score = 0
        self.alive = True
        self.brain = brain if brain else NeuralNet()
        # Couleur
        if color is None:
            self.color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )
        else:
            self.color = color

    def on_ground(self):
        return self.y >= GROUND_Y - self.size - 1

    def jump(self):
        if self.on_ground():
            self.velocityY = -18

    def update(self, obstacles, speed):
        if not self.alive:
            return
        # Gravité
        self.velocityY += self.gravity
        self.y += self.velocityY
        if self.y > GROUND_Y - self.size:
            self.y = GROUND_Y - self.size
            self.velocityY = 0

        # Score = survie
        self.score += 1

        # Trouver le prochain obstacle devant
        next_obs = None
        for obs in obstacles:
            if obs.x + obs.w > self.x:  # encore devant (bord droit de l'obstacle > x dino)
                next_obs = obs
                break

        # Décision IA
        if next_obs:
            inputs = np.array([
                (next_obs.x - self.x) / GAME_WIDTH,  # distance normalisée sur zone de jeu
                next_obs.h / GAME_HEIGHT,            # hauteur obstacle
                speed / 20.0,                        # vitesse normalisée ~ [0,1]
                self.y / GAME_HEIGHT                 # hauteur dino normalisée
            ], dtype=np.float32)
            decision = self.brain.predict(inputs)
            if decision > 0.5:
                self.jump()

    def draw(self, win):
        if self.alive:
            pygame.draw.rect(win, self.color, (self.x, self.y, self.size, self.size))

    def hits(self, obstacle):
        dino_rect = pygame.Rect(self.x, self.y, self.size, self.size)
        obs_rect = pygame.Rect(obstacle.x, obstacle.y, obstacle.w, obstacle.h)
        return dino_rect.colliderect(obs_rect)

# --- Classe Obstacle ---
class Obstacle:
    def __init__(self):
        self.w = random.randint(20, 50)
        self.h = random.randint(30, 70)
        self.x = GAME_WIDTH  # *** Correction: spawn au bord droit de la zone de jeu ***
        self.y = GROUND_Y - self.h

    def update(self, speed):
        self.x -= speed

    def draw(self, win):
        pygame.draw.rect(win, BLACK, (self.x, self.y, self.w, self.h))

    def is_offscreen(self):
        return self.x < -self.w

# --- Évolution ---
def evolve(population, elitism=2, top_k=10, mut_rate=0.1, mut_scale=0.5):
    # Trier meilleurs -> pires
    population.sort(key=lambda d: d.score, reverse=True)
    best = population[:top_k]

    # *** Élites conservés inchangés ***
    new_population = []
    for i in range(min(elitism, len(population))):
        champion = population[i]
        # clone exact (on garde son cerveau et sa couleur)
        clone = Dinosaur(brain=champion.brain, color=champion.color)
        new_population.append(clone)

    # Reproduction jusqu'à remplir la population
    need = len(population) - len(new_population)
    for _ in range(need):
        parent1, parent2 = random.sample(best, 2)
        child_brain = parent1.brain.crossover(parent2.brain) # On prend aléatoirement poids/biais de deux parents dans le top pour le crossover
        child_brain.mutate(rate=mut_rate, scale=mut_scale)
        new_population.append(Dinosaur(child_brain, color=parent1.color))
    return new_population

# --- Tracé des scores ---
def plot_scores():
    if len(score_history) == 0:
        return None
    fig, ax = plt.subplots(figsize=(4, 3))
    x_vals = list(range(1, len(score_history) + 1))
    ax.plot(x_vals, score_history, marker="o")
    ax.set_xlabel("Génération")
    ax.set_ylabel("Score max")
    ax.set_title("Évolution du score")
    ax.set_xlim(1, max(1, len(score_history)))
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = pygame.image.load(buf, "plot.png")
    plt.close(fig)
    return img

# --- Boucle principale ---
def main():
    global GAME_SPEED
    clock = pygame.time.Clock()
    population_size = 50
    dinos = [Dinosaur() for _ in range(population_size)]
    generation = 1
    GAME_SPEED = 6
    best_score_all_time = 0
    progress_img = None

    while True:
        obstacles = []
        running = True
        min_obstacle_gap = 250  # distance minimale entre obstacles (en px)

        while running:
            WIN.fill(WHITE)
            # fond du graphe à droite
            pygame.draw.rect(WIN, (230, 230, 230), (GAME_WIDTH, 0, GRAPH_WIDTH, GAME_HEIGHT))
            clock.tick(60)

            # Événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Génération obstacles
            if not obstacles or (obstacles[-1].x < GAME_WIDTH - min_obstacle_gap):
                if random.random() < 0.02:  # ~1/50 par frame quand l'écart min est respecté
                    obstacles.append(Obstacle())

            # Update dinos
            alive_count = 0
            best_dino = None
            for d in dinos:
                if d.alive:
                    d.update(obstacles, GAME_SPEED)
                    d.draw(WIN)
                    alive_count += 1
                    if best_dino is None or d.score > best_dino.score:
                        best_dino = d
                    for obs in obstacles:
                        if d.hits(obs):
                            d.alive = False

            # Surbrillance du champion en cours
            if best_dino and best_dino.alive:
                pygame.draw.rect(WIN, (255, 0, 0),
                                 (best_dino.x-3, best_dino.y-3,
                                  best_dino.size+6, best_dino.size+6), 2)

            # Update obstacles
            for obs in obstacles[:]:
                obs.update(GAME_SPEED)
                obs.draw(WIN)
                if obs.is_offscreen():
                    obstacles.remove(obs)

            # Sol
            pygame.draw.line(WIN, BLACK, (0, GROUND_Y), (GAME_WIDTH, GROUND_Y), 2)

            # Infos
            if best_dino:
                best_score_all_time = max(best_score_all_time, best_dino.score)

            GAME_SPEED += 0.002
            text = FONT.render(f"Gen: {generation}  Alive: {alive_count}", True, BLACK)
            WIN.blit(text, (20, 20))

            if best_dino:
                info = FONT.render(
                    f"Best Gen: {best_dino.score}  Best All: {best_score_all_time}",
                    True, best_dino.color if hasattr(best_dino, "color") else BLACK
                )
                WIN.blit(info, (20, 50))

            # Graphe
            if progress_img:
                img_rect = progress_img.get_rect(center=(GAME_WIDTH + GRAPH_WIDTH//2, GAME_HEIGHT//2))
                WIN.blit(progress_img, img_rect)

            pygame.display.flip()

            # Fin de génération
            if alive_count == 0:
                scores = [d.score for d in dinos]
                score_history.append(max(scores))
                progress_img = plot_scores()
                # *** Élites + reproduction ***
                dinos = evolve(dinos, elitism=2, top_k=10, mut_rate=0.1, mut_scale=0.5)
                generation += 1
                GAME_SPEED = 6
                running = False

if __name__ == "__main__":
    main()
