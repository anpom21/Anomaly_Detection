import pygame
import random
import sys
import time
from read_bpm import start_heart_rate_stream, bpm_data
import read_bpm


def bpm_to_speed(bpm):
    # Max speed 6.5
    # Min speed 3.0

    speed = 3  # Max speed
    if bpm < 50:
        speed = 6.5
    elif bpm < 150:
        speed = 6.5 - bpm / 42.8

    # Convert BPM to heart rate (example conversion, adjust as needed)
    return speed  # Example conversion


# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Bird setup
bird_speed = 4
bird_width = 30
bird_height = 30
bird_rect = pygame.Rect(50, HEIGHT // 2, bird_width, bird_height)

# Pipe setup
pipe_width = 60
pipe_gap = 60
pipe_height = random.randint(150, 450)
pipe_x = WIDTH
pipe_speed = 3

# Score
font = pygame.font.SysFont(None, 48)
score = 0

# Heart rate
font = pygame.font.SysFont(None, 48)
heart_rate = 60  # Example heart rate

# Game loop
running = True
first_run = True

# Start heart rate stream


start_heart_rate_stream()
while running:
    clock.tick(60)  # 60 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Key press detection
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        bird_rect.y -= bird_speed
    if keys[pygame.K_DOWN]:
        bird_rect.y += bird_speed

    # Pipes
    if first_run:
        pipe_x = WIDTH-pipe_width

    pipe_x -= pipe_speed
    if pipe_x < -pipe_width:
        pipe_x = WIDTH
        pipe_height = random.randint(150, 450)
        score += 1

    top_pipe = pygame.Rect(pipe_x, 0, pipe_width, pipe_height)
    bottom_pipe = pygame.Rect(
        pipe_x, pipe_height + pipe_gap, pipe_width, HEIGHT - pipe_height - pipe_gap)

    # Collision detection
    if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe) or bird_rect.top <= 0 or bird_rect.bottom >= HEIGHT:
        print(f"Game Over! Final Score: {score}")
        pygame.quit()
        sys.exit()

    # Drawing
    screen.fill((135, 206, 235))  # Sky blue background
    pygame.draw.rect(screen, (255, 255, 0), bird_rect)  # Bird - yellow
    pygame.draw.rect(screen, (34, 139, 34), top_pipe)   # Top pipe - green
    pygame.draw.rect(screen, (34, 139, 34), bottom_pipe)  # Bottom pipe - green

    # Score display
    score_text = font.render(str(score), True, (255, 255, 255))
    heart_rate_text = font.render(
        str(read_bpm.bpm_data), True, (255, 0, 0))
    screen.blit(heart_rate_text, (10, 10))
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

    pygame.display.update()
    if first_run:
        time.sleep(1)
        first_run = False
