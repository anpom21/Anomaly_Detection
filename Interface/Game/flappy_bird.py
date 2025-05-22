# ---------------------------------------------------------------------------- #
#                                    Import                                    #
# ---------------------------------------------------------------------------- #
import pygame
import random
import sys
import time
from Game.read_bpm import start_heart_rate_stream, bpm_data
import Game.read_bpm
import yaml
import os

# ---------------------------------------------------------------------------- #
#                                   Functions                                  #
# ---------------------------------------------------------------------------- #
clock = pygame.time.Clock()
working_directory = os.path.dirname(os.path.abspath(__file__))


def wait_for_start_zone(bird_rect, background_img, drone_img, screen, font):
    global clock
    start_zone = pygame.Rect(40, 40, 120, 30)
    bird_velocity = 0

    while True:
        clock.tick(60)
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # sys.exit()

        # Move bird
        bird_velocity = update_bird_velocity(bird_velocity, keys)
        bird_rect.y += bird_velocity

        # Check for collision with start zone
        if bird_rect.colliderect(start_zone):
            return bird_rect  # exit to main game loop

        # Drawing
        screen.blit(background_img, (0, 0))
        screen.blit(drone_img, bird_rect)  # Bird - yellow
        pygame.draw.rect(screen, (255, 0, 0), start_zone)
        start_text = font.render("START", True, (255, 255, 255))
        screen.blit(start_text, (start_zone.centerx - start_text.get_width() // 2,
                                 start_zone.centery - start_text.get_height() // 2))

        pygame.display.update()


def game_over(bird_rect, score, drone_img, screen, font, background_img, background_x, clock):
    # Define start zone as restart trigger
    start_zone = pygame.Rect(40, 150, 300, 30)
    bird_velocity = 0
    # Load gameover image
    gameover_img = pygame.image.load(os.path.join(working_directory, "game_over.png")).convert_alpha()
    gameover_img = pygame.transform.scale(gameover_img, (186, 110))
    gameover_pos = (screen.get_width() // 2 - 100, 10)

    score_bacground_img = pygame.image.load(os.path.join(working_directory, "score_background.png")).convert_alpha()
    score_bacground_img = pygame.transform.scale(score_bacground_img, (186, 110))
    # Shape of the score background

    # Load or initialize high score
    try:
        with open(os.path.join(working_directory, "high_score.yaml"), "r") as file:
            high_score = yaml.safe_load(file) or 0
    except FileNotFoundError:
        high_score = 0
        # Create the file if it doesn't exist
        with open(os.path.join(working_directory, "high_score.yaml"), "w") as file:
            yaml.dump(high_score, file)

    # Update high score if needed
    if score > high_score:
        high_score = score
        with open(os.path.join(working_directory, "high_score.yaml"), "w") as file:
            yaml.dump(high_score, file)

    # Game over loop
    while True:
        clock.tick(60)
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # sys.exit()

        # Move bird so it can "fly into" start zone to restart
        bird_velocity = update_bird_velocity(bird_velocity, keys)
        bird_rect.y += bird_velocity

        # Check if player wants to restart
        if bird_rect.colliderect(start_zone):
            return bird_rect, background_x  # Reset to caller with new position

        # Draw everything
        screen.blit(background_img, (background_x, 0))
        screen.blit(background_img, (background_x + 587, 0))
        screen.blit(drone_img, bird_rect)

        pygame.draw.rect(screen, (255, 0, 0), start_zone)
        font = pygame.font.SysFont(None, 48)
        restart_text = font.render(
            "Fly Here to Restart", True, (255, 255, 255))
        screen.blit(restart_text, (start_zone.centerx - restart_text.get_width() // 2,
                                   start_zone.centery - restart_text.get_height() // 2))

        screen.blit(gameover_img, (screen.get_width() // 2-186//2, 10))
        # --- Score display --- #
        font = pygame.font.SysFont(None, 35)
        screen.blit(score_bacground_img,
                    (screen.get_width() // 2-186//2, 200))

        # Score
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (screen.get_width() //
                    2 - score_text.get_width() // 2, 215))
        # High score
        high_score_text = font.render(
            f"High Score: {high_score}", True, (255, 255, 255))
        screen.blit(high_score_text, (screen.get_width() //
                    2 - high_score_text.get_width() // 2, 260))

        pygame.display.update()


def bpm_to_speed(bpm):
    # Max speed 6.5
    # Default speed 4.0
    # Min speed 3.0

    speed_scaling = 1
    delta_bpm = Game.read_bpm.initial_bpm - bpm
    speed = 4.0 + delta_bpm * speed_scaling

    # Ensure speed is within bounds
    if speed < 1:
        speed = 1
    elif speed > 6.5:
        speed = 6.5 * 0.2 * delta_bpm

    # Convert BPM to heart rate (example conversion, adjust as needed)
    return speed  # Example conversion


def update_bird_velocity(velocity, keys, acceleration=0.5, max_speed=6, friction=0.3):
    if keys[pygame.K_UP]:
        velocity -= acceleration
    elif keys[pygame.K_DOWN]:
        velocity += acceleration
    else:
        velocity *= (1 - friction)  # Apply friction when no key is pressed
        velocity = 0
    # Clamp the velocity
    if velocity > max_speed:
        velocity = max_speed
    elif velocity < -max_speed:
        velocity = -max_speed

    return velocity


def read_position(bird_rect):
    # Read the position from the file
    keys = pygame.key.get_pressed()
    acceleration = 1
    max_speed = 6.5
    friction = 10
    bird_speed = update_bird_velocity(
        bird_speed, keys, acceleration, max_speed, friction)
    bird_position = bird_rect.y + bird_speed

    return bird_position


# ---------------------------------------------------------------------------- #
#                                Intialize game                                #
# ---------------------------------------------------------------------------- #
def run_game():
    # Initialize pygame
    pygame.init()

    # Screen settings
    WIDTH, HEIGHT = 400, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Background setup
    background_width = 587
    background_img = pygame.image.load(os.path.join(working_directory, "background_looping.png")).convert()
    background_img = pygame.transform.scale(
        background_img, (background_width, HEIGHT))  # 587x600
    background_start_x = 0
    background_x = background_start_x
    scroll_speed = 1.5  # Tune as you like

    # Bird setup (pos 0-565)
    bird_speed = 4
    bird_width = 35
    bird_height = 35
    drone_img = pygame.image.load(os.path.join(working_directory, "drone_white.png")).convert_alpha()
    # Scale the image to fit the bird size
    drone_img = pygame.transform.scale(drone_img, (bird_width, bird_height))
    bird_rect = pygame.Rect(50, HEIGHT // 2, bird_width, bird_height)

    # Pipe setup
    pipe_width = 60
    pipe_gap = 60
    pipe_height = random.randint(150, 450)
    pipe_x = WIDTH
    pipe_speed = 3
    laser_img = pygame.image.load(os.path.join(working_directory, "laser4.png")).convert_alpha()
    laser_width, laser_height = laser_img.get_size()
    top_pipe = pygame.Rect(pipe_x, 0, pipe_width, pipe_height)
    bottom_pipe = pygame.Rect(
        pipe_x, pipe_height + pipe_gap, pipe_width, HEIGHT - pipe_height - pipe_gap)

    # Score
    font = pygame.font.SysFont(None, 48)
    score = 0

    # Heart rate
    font = pygame.font.SysFont(None, 48)
    heart_rate = 60  # Example heart rate

    # Game loop
    running = True
    first_run = True

    # Timer
    start_timer = 0

    # Start heart rate stream
    start_heart_rate_stream()

    bird_rect = wait_for_start_zone(
        bird_rect, background_img, drone_img, screen, font)

    # ---------------------------------------------------------------------------- #
    #                               Run the game loop                              #
    # ---------------------------------------------------------------------------- #
    while running:
        clock.tick(60)  # 60 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # sys.exit()

        # ------------------------------------ Bird ---------------------------------- #
        # Bird movement
        keys = pygame.key.get_pressed()
        acceleration = 1
        max_speed = 6.5
        friction = 10
        bird_speed = update_bird_velocity(
            bird_speed, keys, acceleration, max_speed, friction)

        bird_rect.y += bird_speed
        # ----------------------------------- Pipes ---------------------------------- #
        if first_run:
            pipe_x = WIDTH-pipe_width

        # Adjust pipe speed based on heart rate
        pipe_x -= bpm_to_speed(Game.read_bpm.bpm_data)
        if pipe_x < -pipe_width:
            pipe_x = WIDTH
            pipe_height = random.randint(150, 450)
            score += 1

        top_pipe = pygame.Rect(pipe_x, 0, pipe_width, pipe_height)
        bottom_pipe = pygame.Rect(
            pipe_x, pipe_height + pipe_gap, pipe_width, HEIGHT - pipe_height - pipe_gap)
        # ---------------------------- Collision detection --------------------------- #
        if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe) or bird_rect.top <= 0 or bird_rect.bottom >= HEIGHT:
            print(f"Game Over! Final Score: {score}")
            # Reset game state
            bird_speed = 0
            pipe_x = WIDTH
            bird_rect, background_start_x = game_over(
                bird_rect, score, drone_img, screen, font, background_img, background_x, clock)
            start_timer = 0
            first_run = True
            score = 0
            # bird_rect = wait_for_start_zone(bird_rect, drone_img, screen, font)
        # -------------------------------- Background -------------------------------- #
        # Update scroll position
        background_x -= pipe_speed/4 * 1.5  # scroll_speed
        if background_x <= -background_width:
            background_x += background_width

        if first_run:
            background_x = background_start_x

        # Draw two copies to handle wrap-around
        screen.blit(background_img, (background_x, 0))
        screen.blit(background_img, (background_x + background_width, 0))

        # ---------------------------------- Drawing --------------------------------- #

        screen.blit(drone_img, bird_rect)  # Bird - yellow
        # For top laser (positioned based on pipe_height)
        top_laser_y = pipe_height - laser_height
        screen.blit(laser_img, (pipe_x, top_laser_y))

        # For bottom laser
        bottom_laser_y = pipe_height + pipe_gap
        screen.blit(laser_img, (pipe_x, bottom_laser_y))

        # ------------------------------- Score display ------------------------------ #
        score_text = font.render(str(score), True, (255, 255, 255))
        heart_rate_text = font.render(
            str(Game.read_bpm.bpm_data), True, (255, 0, 0))
        screen.blit(heart_rate_text, (10, 10))
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

        if start_timer == 0:
            start_timer = pygame.time.get_ticks()
        elif pygame.time.get_ticks() - start_timer > 1000:
            first_run = False

        pygame.display.update()


if __name__ == "__main__":

    run_game()
