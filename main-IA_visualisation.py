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
GAME_WIDTH, GAME_HEIGHT = 800, 400  # zone du jeu
GRAPH_WIDTH = 400
WIDTH, HEIGHT = GAME_WIDTH + GRAPH_WIDTH, GAME_HEIGHT
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino AI - Algorithme Génétique")

WHITE = (247, 247, 247)
BLACK = (50, 50, 50)
GROUND_Y = GAME_HEIGHT - 40
FONT = pygame.font.SysFont("Courier New", 20)

# Stocker les stats
score_history = []


# --- Réseau de neurones simple ---
class NeuralNet:
    def __init__(self):
        # 4 entrées -> 1 couche cachée (6 neurones) -> 1 sortie
        self.w1 = np.random.randn(4, 6)  # poids entrée -> cachée
        self.w2 = np.random.randn(6, 1)  # poids cachée -> sortie

    def predict(self, inputs):
        hidden = np.tanh(np.dot(inputs, self.w1))  # couche cachée
        output = 1 / (1 + np.exp(-np.dot(hidden, self.w2)))  # sigmoid
        return output[0]

    def crossover(self, partner):
        child = NeuralNet()
        # mélange les poids des parents
        child.w1 = np.where(np.random.rand(*self.w1.shape) > 0.5, self.w1, partner.w1)
        child.w2 = np.where(np.random.rand(*self.w2.shape) > 0.5, self.w2, partner.w2)
        return child

    def mutate(self, rate=0.1):
        # ajoute de petites perturbations
        mutation_mask1 = np.random.rand(*self.w1.shape) < rate
        mutation_mask2 = np.random.rand(*self.w2.shape) < rate
        self.w1 += mutation_mask1 * np.random.randn(*self.w1.shape) * 0.5
        self.w2 += mutation_mask2 * np.random.randn(*self.w2.shape) * 0.5


# --- Classe Dinosaur avec IA ---
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

        # Couleur unique par dino
        if color is None:
            self.color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )
        else:
            self.color = color

    def jump(self):
        if self.y == GROUND_Y - self.size:
            self.velocityY = -18

    def update(self, obstacles, speed):
        if not self.alive:
            return

        # appliquer gravité
        self.velocityY += self.gravity
        self.y += self.velocityY
        if self.y > GROUND_Y - self.size:
            self.y = GROUND_Y - self.size
            self.velocityY = 0

        # incrémenter score
        self.score += 1

        # prendre le premier obstacle devant
        next_obs = None
        for obs in obstacles:
            if obs.x + obs.w > self.x:
                next_obs = obs
                break

        if next_obs:
            # préparer les entrées du réseau
            inputs = np.array([
                (next_obs.x - self.x) / WIDTH,  # distance
                next_obs.h / HEIGHT,           # hauteur
                speed / 20,                    # vitesse normalisée
                self.y / HEIGHT                # position du dino
            ])
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
        self.x = WIDTH
        self.y = GROUND_Y - self.h

    def update(self, speed):
        self.x -= speed

    def draw(self, win):
        pygame.draw.rect(win, BLACK, (self.x, self.y, self.w, self.h))

    def is_offscreen(self):
        return self.x < -self.w


# --- Génération et évolution ---
def evolve(population):
    # trier par score
    population.sort(key=lambda d: d.score, reverse=True)
    best = population[:10]  # les 10 meilleurs
    new_population = []

    # reproduire
    for _ in range(len(population)):
        parent1, parent2 = random.sample(best, 2)
        child_brain = parent1.brain.crossover(parent2.brain)
        child_brain.mutate(0.1)
        new_population.append(Dinosaur(child_brain, color=parent1.color))

    return new_population


# --- Fonction de visualisation ---
def plot_scores():
    if len(score_history) == 0:
        return None

    fig, ax = plt.subplots(figsize=(4, 3))
    x_vals = list(range(1, len(score_history) + 1))
    ax.plot(x_vals, score_history, color="blue", marker="o")
    ax.set_xlabel("Génération")
    ax.set_ylabel("Score max")
    ax.set_title("Évolution du score")
    ax.set_xlim(1, max(1, len(score_history)))  # toujours commencer à 1
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
    best_score_all_time = 0  # meilleur score global
    progress_img = None

    while True:
        obstacles = []
        running = True
        min_obstacle_gap = 250  # distance minimale entre obstacles

        while running:
            WIN.fill(WHITE)
            # fond du graphe à droite
            pygame.draw.rect(WIN, (230, 230, 230), (GAME_WIDTH, 0, GRAPH_WIDTH, GAME_HEIGHT))
            clock.tick(60)

            # --- événements (fermer fenêtre) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # --- générer obstacles ---
            if not obstacles or (obstacles[-1].x < WIDTH - min_obstacle_gap):
                if random.random() < 0.02:
                    obstacles.append(Obstacle())

            # --- update dinos ---
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

            # contour autour du champion actuel
            if best_dino and best_dino.alive:
                pygame.draw.rect(WIN, (255, 0, 0),
                                 (best_dino.x-3, best_dino.y-3,
                                  best_dino.size+6, best_dino.size+6), 2)

            # --- update obstacles ---
            for obs in obstacles[:]:
                obs.update(GAME_SPEED)
                obs.draw(WIN)
                if obs.is_offscreen():
                    obstacles.remove(obs)

            # --- sol ---
            pygame.draw.line(WIN, BLACK, (0, GROUND_Y), (GAME_WIDTH, GROUND_Y), 2)

            # --- affichage infos ---
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

            # --- afficher la courbe à droite ---
            if progress_img:
                # centrer le graphe dans la zone de droite
                img_rect = progress_img.get_rect(center=(GAME_WIDTH + GRAPH_WIDTH//2, GAME_HEIGHT//2))
                WIN.blit(progress_img, img_rect)

            pygame.display.flip()

            # --- si tous morts -> nouvelle génération ---
            if alive_count == 0:
                scores = [d.score for d in dinos]
                score_history.append(max(scores))

                # mettre à jour le graphe
                progress_img = plot_scores()

                dinos = evolve(dinos)
                generation += 1
                GAME_SPEED = 6
                running = False


if __name__ == "__main__":
    main()