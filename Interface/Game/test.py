import pygame
import os

pygame.init()                                  # this initializes the font module too

# … your setup code (screen, scaling, etc.) …
scaling = 1.0  # Adjust this based on your screen size or scaling factor
# Example screen size, adjust as needed
screen = pygame.display.set_mode((800, 600))
font_path = os.path.join("Game", "Minecraft.ttf")
font_size = int(48 * scaling)
# note: pygame.font, not pixel_font
pixel_font = pygame.font.Font(font_path, font_size)

# Later, when you need to draw:
text_surf = pixel_font.render("Score: 123", False, (255, 255, 255))
screen.blit(text_surf, (100, 100))
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Update the display
    pygame.display.flip()
