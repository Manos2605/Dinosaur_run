import pygame
import random
import sys
import numpy as np

# --- Initialisation ---
pygame.init()
WIDTH, HEIGHT = 800, 400
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino AI - Algorithme Génétique")

WHITE = (247, 247, 247)
BLACK = (50, 50, 50)
GROUND_Y = HEIGHT - 40
FONT = pygame.font.SysFont("Courier New", 20)

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

        # Couleur unique par dino (hérite du parent si donnée)
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
        # l’enfant hérite de la couleur du parent1 pour suivre les lignées
        new_population.append(Dinosaur(child_brain, color=parent1.color))

    return new_population

# --- Boucle principale ---
def main():
    global GAME_SPEED
    clock = pygame.time.Clock()
    population_size = 50
    dinos = [Dinosaur() for _ in range(population_size)]
    generation = 1
    GAME_SPEED = 6

    while True:
        obstacles = []
        score = 0
        running = True
        min_obstacle_gap = 250  # distance minimale entre obstacles
        last_obstacle_x = WIDTH

        while running:
            clock.tick(60)
            WIN.fill(WHITE)

            # --- événements (fermer fenêtre) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # --- générer obstacles avec espacement minimal ---
            if not obstacles or (obstacles[-1].x < WIDTH - min_obstacle_gap):
                # Probabilité d'apparition, ajustable
                if random.random() < 0.02:
                    obstacles.append(Obstacle())

            # --- update dinos ---
            alive_count = 0
            for d in dinos:
                if d.alive:
                    d.update(obstacles, GAME_SPEED)
                    d.draw(WIN)
                    alive_count += 1
                    for obs in obstacles:
                        if d.hits(obs):
                            d.alive = False

            # --- update obstacles ---
            for obs in obstacles[:]:
                obs.update(GAME_SPEED)
                obs.draw(WIN)
                if obs.is_offscreen():
                    obstacles.remove(obs)

            # --- sol ---
            pygame.draw.line(WIN, BLACK, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)

            # --- affichage infos ---
            score += 1
            GAME_SPEED += 0.002
            text = FONT.render(f"Gen: {generation}  Alive: {alive_count}", True, BLACK)
            WIN.blit(text, (20, 20))

            pygame.display.flip()

            # --- si tous morts -> nouvelle génération ---
            if alive_count == 0:
                dinos = evolve(dinos)
                generation += 1
                GAME_SPEED = 6
                running = False

if __name__ == "__main__":
    main()
