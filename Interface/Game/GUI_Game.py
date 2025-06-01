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
import math
import os

from serial_read_MR_together import SerialReaderThread

# ---------------------------------------------------------------------------- #
#                                   Functions                                  #
# ---------------------------------------------------------------------------- #
clock = pygame.time.Clock()
working_directory = os.path.dirname(os.path.abspath(__file__))
scaling = 1.45
# Adjust COM port and baudrate as needed
debugging = False  # Set to True for debugging without pressure sensor

min_pos = 181
max_pos = 214

phase = 0
period = 0
sine_val = 0
start_time = time.time()
points = None
pygame.init()
font_path = os.path.join("Game", "Minecraft.ttf")
pixel_font = pygame.font.Font(font_path, int(30*scaling))


def start_menu(bird_rect, background_img, drone_img, screen, font, HEIGHT, WIDTH, thread):
    global debugging, scaling, clock, points, pixel_font, min_pos

    start_zone = pygame.Rect(40*scaling, 40*scaling, 120*scaling, 30*scaling)
    bird_velocity = 0

    start = False

    while not start:
        clock.tick(60)
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Stopping...")
                pygame.quit()
                thread.kill()  # Ensure thread is stopped and joined
                del thread  # Ensure thread is deleted
                sys.exit()

        # Get points from pressure sensor
        if points is None and not debugging:
            pressure = thread.get_pressure()
            if pressure is not None:
                points = round(pressure * 4)
            else:
                print("No pressure data available")
        # Move bird
        bird_velocity = update_bird_velocity(bird_velocity, keys)
        bird_rect.y = read_position(bird_rect, thread)

        # Check for the user pressing 'SPACE' to start
        if keys[pygame.K_SPACE]:
            min_pos = thread.get_position()
            start = True

        # Drawing
        screen.blit(background_img, (0, 0))
        screen.blit(drone_img, bird_rect)  # Bird - yellow
        # Change font size based
        pixel_font = pygame.font.Font(
            "Game/Minecraft.ttf", int(30*scaling))
        start_text = pixel_font.render("START", True, (255, 255, 255))
        line_1 = pixel_font.render(
            "Extend your legs fully", True, (255, 255, 255))
        line_2 = pixel_font.render(
            "and press SPACE to start", True, (255, 255, 255))

        screen.blit(line_1, (WIDTH // 2 - line_1.get_width() // 2,
                             100*scaling))
        screen.blit(line_2, (WIDTH // 2 - line_2.get_width() // 2,
                             100*scaling + 10*scaling + 20*scaling))

        pygame.display.update()

    return bird_rect


def wait_for_start_zone(bird_rect, background_img, drone_img, screen, font, thread):
    global debugging, scaling, clock, points, pixel_font

    start_zone = pygame.Rect(40*scaling, 40*scaling, 120*scaling, 30*scaling)
    bird_velocity = 0

    while True:
        clock.tick(60)
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Stopping...")
                pygame.quit()
                thread.kill()  # Ensure thread is stopped and joined
                sys.exit()

        # Get points from pressure sensor
        if points is None and not debugging:
            pressure = thread.get_pressure()
            if pressure is not None:
                points = round(pressure * 4)
            else:
                print("No pressure data available")
        # Move bird
        bird_velocity = update_bird_velocity(bird_velocity, keys)
        bird_rect.y = read_position(bird_rect, thread)

        # Check for collision with start zone
        if bird_rect.colliderect(start_zone):
            return bird_rect  # exit to main game loop

        # Drawing
        screen.blit(background_img, (0, 0))
        screen.blit(drone_img, bird_rect)  # Bird - yellow
        pygame.draw.rect(screen, (255, 0, 0), start_zone)
        # Change font size based
        pixel_font = pygame.font.Font(
            "Game/Minecraft.ttf", int(30*scaling))
        start_text = pixel_font.render("START", True, (255, 255, 255))
        screen.blit(start_text, (start_zone.centerx - start_text.get_width() // 2,
                                 start_zone.centery - start_text.get_height() // 2+3*scaling))

        pygame.display.update()


def game_over(bird_rect, score, drone_img, screen, font, background_img, background_x, clock, thread):
    global debugging, scaling, pixel_font
    # Define start zone as restart trigger
    start_zone = pygame.Rect(40*scaling, 150*scaling, 300*scaling, 30*scaling)

    green_box = pygame.Rect(
        40*scaling, 300*scaling, 30*scaling, 30*scaling)
    green_box_color = (0, 255, 0, 50)  # Green color for the box
    bird_velocity = 0
    # Load gameover image
    gameover_img = pygame.image.load(os.path.join(
        working_directory, "game_over.png")).convert_alpha()
    gameover_img = pygame.transform.scale(
        gameover_img, (186*scaling, 110*scaling))
    # gameover_pos = (screen.get_width() // 2 - 100, 10)

    score_bacground_img = pygame.image.load(os.path.join(
        working_directory, "score_background.png")).convert_alpha()
    score_bacground_img = pygame.transform.scale(
        score_bacground_img, (186*scaling, 110*scaling))
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
    if score > high_score and not debugging:
        high_score = score
        with open(os.path.join(working_directory, "high_score.yaml"), "w") as file:
            yaml.dump(high_score, file)

    # Game over loop
    first_run = True
    while True:
        # print("Get position", bird_rect.y)
        clock.tick(60)
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Stopping...")

                pygame.quit()

                thread.kill()  # Ensure thread is stopped and joined
                sys.exit()

        # Move bird so it can "fly into" start zone to restart
        bird_velocity = update_bird_velocity(bird_velocity, keys)
        bird_rect.y = read_position(bird_rect, thread)

        if bird_rect.y > 300 * scaling:
            first_run = False

        # Check if player wants to restart
        if bird_rect.colliderect(start_zone) and not first_run:
            return bird_rect, background_x  # Reset to caller with new position

        # Draw everything
        screen.blit(background_img, (background_x, 0))
        screen.blit(background_img, (background_x + 587*scaling, 0))
        screen.blit(drone_img, bird_rect)

        pixel_font = pygame.font.Font(
            "Game/Minecraft.ttf", int(30*scaling))
        # FONTSIZE
        restart_text = pixel_font.render(
            "Fly here to Restart", True, (255, 255, 255))

        screen.blit(gameover_img, (screen.get_width() //
                    2-186*scaling//2, 10*scaling))
        # --- Score display --- #
        pixel_font = pygame.font.Font(
            "Game/Minecraft.ttf", int(23*scaling))
        score_text = pixel_font.render(
            f"Score: {score}", True, (255, 255, 255))
        high_score_text = pixel_font.render(
            f"High Score: {high_score}", True, (255, 255, 255))
        screen.blit(score_bacground_img,
                    (screen.get_width() // 2-186*scaling//2, 200*scaling))

        # Score
        screen.blit(score_text, (screen.get_width() //
                    2 - score_text.get_width() // 2, 219*scaling))
        # High score
        screen.blit(high_score_text, (screen.get_width() //
                    2 - high_score_text.get_width() // 2, 264*scaling))

        if not first_run:
            pygame.draw.rect(screen, (255, 0, 0), start_zone)
            screen.blit(restart_text, (start_zone.centerx - restart_text.get_width() // 2,
                                       start_zone.centery - restart_text.get_height() // 2+3*scaling))

        pygame.display.update()


def bpm_to_speed(bpm):
    # Max speed 6.5
    # Default speed 4.0
    # Min speed 3.0

    speed_scaling = 1
    delta_bpm = Game.read_bpm.initial_bpm - bpm  # Changed
    speed = 4.0 + delta_bpm * speed_scaling

    # Ensure speed is within bounds
    if speed < 1:
        speed = 1
    elif speed > 6.5:
        speed = 6.5 * 0.2 * delta_bpm

    # Convert BPM to heart rate (example conversion, adjust as needed)
    return speed  # Example conversion


def update_bird_velocity(velocity, keys, acceleration=0.5, max_speed=6, friction=0.3):
    velocity = 0
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

    if keys[pygame.K_UP]:
        velocity -= 3
    elif keys[pygame.K_DOWN]:
        velocity += 3
    else:
        velocity = 0

    return velocity


def read_position(bird_rect, thread):
    global debugging, scaling, min_pos, max_pos
    # Read the position from the file
    bird_max = 565 * scaling
    bird_min = 0
    # debugging = True

    if True:  # debugging:
        bird_speed = 4
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            bird_speed = -bird_speed
        elif keys[pygame.K_DOWN]:
            bird_speed = bird_speed
        else:
            bird_speed = 0

        bird_pos = bird_rect.y + bird_speed
    else:
        # Read the position from the serial thread
        bird_pos = thread.get_position()
        if bird_pos is None:
            bird_pos = 0
        # Map the bird_max and bird_min to the min_pos and max_pos
        bird_pos = (bird_pos - min_pos) / (max_pos - min_pos) * \
            (bird_max - bird_min) + bird_min
    return bird_pos


# ---------------------------------------------------------------------------- #
#                                Intialize game                                #
# ---------------------------------------------------------------------------- #
def run_game(thread):
    global debugging, scaling, clock, points, phase, sine_val, pixel_font
    # -------------------------------- Initialize -------------------------------- #
    clock = pygame.time.Clock()
    working_directory = os.path.dirname(os.path.abspath(__file__))
    scaling = 1.45
    # Adjust COM port and baudrate as needed
    debugging = False  # Set to True for debugging without pressure sensor

    min_pos = 181
    max_pos = 214

    phase = 0
    period = 0
    sine_val = 0
    start_time = time.time()
    points = None
    pygame.init()
    font_path = os.path.join("Game", "Minecraft.ttf")
    pixel_font = pygame.font.Font(font_path, int(30*scaling))
    # Get points from pressure sensor

    if debugging:
        points = 4
    elif thread.get_pressure() is not None:
        points = round(thread.get_pressure() * 4)
    else:
        points = None
    print(f"Points from pressure sensor: {points}")
    # Initialize pygame
    pygame.init()

    # Screen settings
    WIDTH, HEIGHT = 400*scaling, 600*scaling
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Background setup
    background_width = 587*scaling
    background_img = pygame.image.load(os.path.join(
        working_directory, "background_looping.png")).convert()
    background_img = pygame.transform.scale(
        background_img, (background_width, HEIGHT))  # 587x600
    background_start_x = 0
    background_x = background_start_x
    scroll_speed = 1.5  # Tune as you like

    # Bird setup (pos 0-565)
    bird_speed = 4
    bird_width = int(35*scaling)
    bird_height = int(35*scaling)
    drone_img = pygame.image.load(os.path.join(
        working_directory, "drone_white.png")).convert_alpha()
    # Scale the image to fit the bird size
    drone_img = pygame.transform.scale(drone_img, (bird_width, bird_height))
    bird_rect = pygame.Rect(50*scaling, HEIGHT // 2, bird_width, bird_height)

    # Pipe setup
    pipe_width = 60*scaling
    pipe_gap = 120*scaling
    old_pipe_passthroughs = 0
    pipe_height = random.randint(int(150*scaling), int(450*scaling))
    pipe_x = WIDTH
    pipe_speed = 3
    laser_img = pygame.image.load(os.path.join(
        working_directory, "laser4.png")).convert_alpha()
    laser_width, laser_height = laser_img.get_size()
    pipe_scaling = 1.2 * pipe_width / laser_width
    laser_img = pygame.transform.scale(
        laser_img, (laser_width*pipe_scaling, laser_height))
    top_pipe = pygame.Rect(pipe_x, 0, pipe_width, pipe_height)
    bottom_pipe = pygame.Rect(
        pipe_x, pipe_height + pipe_gap, pipe_width, HEIGHT - pipe_height - pipe_gap)

    # Score
    pixel_font = pygame.font.Font(
        "Game/Minecraft.ttf", int(30*scaling))
    score = 0

    # Heart rate
    pixel_font = pygame.font.Font(
        "Game/Minecraft.ttf", int(30*scaling))
    heart_rate = 60  # Example heart rate
    # Heart image
    # Load once
    heart_orig = pygame.image.load(os.path.join(
        working_directory, "heart.png")).convert_alpha()
    w0, h0 = heart_orig.get_size()
    # Heart position
    heart_pos = (55, 55)
    # tuning: how much bigger/smaller (e.g. ±20%)
    AMP = 0.06
    start_time = time.time()

    # Game loop
    running = True
    first_run = True

    # Timer
    start_timer = 0

    # Start heart rate stream
    start_heart_rate_stream()

    if debugging:
        bird_rect = wait_for_start_zone(
            bird_rect, background_img, drone_img, screen, pixel_font, thread)
    else:
        bird_rect = start_menu(
            bird_rect, background_img, drone_img, screen, pixel_font, HEIGHT, WIDTH, thread)
    period = 60.0 / max(Game.read_bpm.bpm_data, 1e-2)

    if points is None:
        print("[ERROR] No points available from pressure sensor. Exiting game.")
        pygame.quit()
        thread.kill()
        sys.exit()
    # ---------------------------------------------------------------------------- #
    #                               Run the game loop                              #
    # ---------------------------------------------------------------------------- #
    while running:

        dt = clock.tick(60) / 1000.0   # frame time in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Stopping...")

                pygame.quit()
                thread.kill()  # Ensure thread is stopped and joined
                sys.exit()

        # ------------------------------------ Bird ---------------------------------- #
        # Bird movement
        keys = pygame.key.get_pressed()
        # acceleration = 1
        # max_speed = 6.5
        # friction = 10
        # bird_speed = update_bird_velocity(
        #     bird_speed, keys, acceleration, max_speed, friction)

        bird_rect.y = read_position(bird_rect, thread)
        # ----------------------------------- Pipes ---------------------------------- #
        if first_run:
            pipe_x = WIDTH-pipe_width

        pipe_passthroughs = score // points

        if pipe_passthroughs > old_pipe_passthroughs and pipe_gap > 50*scaling:
            old_pipe_passthroughs = pipe_passthroughs
            pipe_gap -= 8*scaling  # Decrease gap by 5 pixels for each point scored

        # Adjust pipe speed based on heart rate
        pipe_x -= bpm_to_speed(Game.read_bpm.bpm_data)  # Changed
        if pipe_x < -pipe_width:
            pipe_x = WIDTH
            pipe_height = random.randint(150, 450)
            score += points

        top_pipe = pygame.Rect(pipe_x, 0, pipe_width, pipe_height)
        bottom_pipe = pygame.Rect(
            pipe_x, pipe_height + pipe_gap, pipe_width, HEIGHT - pipe_height - pipe_gap)
        # ---------------------------- Collision detection --------------------------- #
        if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe) or bird_rect.top <= -HEIGHT or bird_rect.bottom >= HEIGHT:
            print(f"Game Over! Final Score: {score}")
            # Reset game state
            bird_speed = 0
            pipe_x = WIDTH
            bird_rect, background_start_x = game_over(
                bird_rect, score, drone_img, screen, pixel_font, background_img, background_x, clock, thread)
            start_timer = 0
            first_run = True
            score = 0
            pipe_gap = 120*scaling

            # bird_rect = wait_for_start_zone(bird_rect, drone_img, screen, pixel_font)
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
        # ------------------------------- Heart rate -------------------------------- #
        phase += (2 * math.pi) * (dt / period)

        if sine_val < 0 and math.sin(2*math.pi * t / period) > 0:
            phase -= 2 * math.pi

            # now read a fresh BPM and update period
            bpm = Game.read_bpm.bpm_data
            period = 60.0 / max(bpm, 1e-2)
        t = (time.time() - start_time) % period
        sine_val = math.sin(2*math.pi * t / period)
        scale_factor = 1.0 + AMP * sine_val   # between 1-AMP and 1+AMP
        # print(
        #     f"Current BPM: {Game.read_bpm.bpm_data}, Period: {scale_factor}, Sine Value: {sine_val}")
        new_w = int(w0 * scale_factor)
        new_h = int(h0 * scale_factor)
        heart_scaled = pygame.transform.scale(heart_orig, (new_w, new_h))
        rect = heart_scaled.get_rect(center=heart_pos)
        screen.blit(heart_scaled, rect)

        # ------------------------------- Score display ------------------------------ #
        pixel_font = pygame.font.Font(
            "Game/Minecraft.ttf", int(40*scaling))
        score_text = pixel_font.render(str(score), True, (255, 255, 255))
        pixel_font = pygame.font.Font(
            "Game/Minecraft.ttf", int(30*scaling))
        heart_rate_text = pixel_font.render(
            str(Game.read_bpm.bpm_data), True, (255, 255, 255), )  # Changed
        screen.blit(heart_rate_text, (33, 35))
        screen.blit(score_text, (WIDTH // 2 -
                    score_text.get_width() // 2, 20*scaling))

        if start_timer == 0:
            start_timer = pygame.time.get_ticks()
        elif pygame.time.get_ticks() - start_timer > 1000:
            first_run = False

        pygame.display.update()


if __name__ == "__main__":

    run_game()
