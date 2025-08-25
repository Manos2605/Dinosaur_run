import pygame
import random
import sys

# --- Initialisation ---
pygame.init()
WIDTH, HEIGHT = 800, 400
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Game en Python")

# --- Couleurs ---
WHITE = (247, 247, 247)
BLACK = (50, 50, 50)

# --- Paramètres globaux ---
GROUND_Y = HEIGHT - 40
GAME_SPEED = 6
FONT = pygame.font.SysFont("Courier New", 20)

# --- Classe Dinosaur ---
class Dinosaur:
    def __init__(self):
        self.size = 50
        self.x = 60
        self.y = GROUND_Y - self.size
        self.velocityY = 0
        self.gravity = 0.7

    def jump(self):
        if self.y == GROUND_Y - self.size:  # sauter seulement si au sol
            self.velocityY = -18

    def update(self):
        self.velocityY += self.gravity
        self.y += self.velocityY
        if self.y > GROUND_Y - self.size:
            self.y = GROUND_Y - self.size
            self.velocityY = 0

    def draw(self, win):
        pygame.draw.rect(win, BLACK, (self.x, self.y, self.size, self.size))

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

    def update(self):
        self.x -= GAME_SPEED

    def draw(self, win):
        pygame.draw.rect(win, BLACK, (self.x, self.y, self.w, self.h))

    def is_offscreen(self):
        return self.x < -self.w

# --- Boucle principale ---
def main():
    global GAME_SPEED
    clock = pygame.time.Clock()
    dino = Dinosaur()
    obstacles = []
    score = 0
    game_over = False

    while True:
        clock.tick(60)  # 60 FPS
        WIN.fill(WHITE)

        # --- Gestion des événements ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if game_over:
                        return main()  # restart
                    else:
                        dino.jump()

        # --- Si le jeu est fini ---
        if game_over:
            text = FONT.render("GAME OVER - Appuyez ESPACE pour recommencer", True, BLACK)
            WIN.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2))
            pygame.display.flip()
            continue

        # --- Générer obstacles ---
        if random.randint(0, 100) < 2:
            obstacles.append(Obstacle())

        # --- Update dino ---
        dino.update()
        dino.draw(WIN)

        # --- Update obstacles ---
        for obs in obstacles[:]:
            obs.update()
            obs.draw(WIN)
            if dino.hits(obs):
                game_over = True
            if obs.is_offscreen():
                obstacles.remove(obs)

        # --- Sol ---
        pygame.draw.line(WIN, BLACK, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)

        # --- Score ---
        score += 1
        score_text = FONT.render(f"Score: {score // 10}", True, BLACK)
        WIN.blit(score_text, (WIDTH - 150, 20))

        # --- Vitesse qui augmente petit à petit ---
        GAME_SPEED += 0.002

        pygame.display.flip()

if __name__ == "__main__":
    main()
