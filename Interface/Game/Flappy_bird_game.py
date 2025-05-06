import pygame
import random
import time
from Game.read_bpm import start_heart_rate_stream, bpm_data

# Initialize all pygame modules, including font
pygame.init()

class PygameGame:
    """
    Embedded Flappy-BPM game. Use by calling step() in a loop, or run() standalone.
    """
    def __init__(self, width=400, height=600, fps=30):
        # Initialize display and font subsystems explicitly
        pygame.display.init()
        pygame.font.init()
        # Game parameters
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = None
        self.clock = None
        # Bird parameters
        self.bird_width = 30
        self.bird_height = 30
        self.bird_speed = 4
        self.bird_rect = None
        # Pipe parameters
        self.pipe_width = 60
        self.pipe_gap = 60
        self.pipe_speed = 3
        self.pipe_height = None
        self.pipe_x = None
        # Score
        self.score = 0
        # Running flag
        self.running = False
        # Start heart rate stream
        start_heart_rate_stream()
        # Initialize game state
        self._reset_game_state()

    def _reset_game_state(self):
        # Initialize bird rect
        self.bird_rect = pygame.Rect(50, self.height // 2, self.bird_width, self.bird_height)
        # Initialize pipe
        self._reset_pipe()
        # Reset score
        self.score = 0
        # Initialize clock
        self.clock = pygame.time.Clock()

    def _reset_pipe(self):
        self.pipe_height = random.randint(150, 450)
        self.pipe_x = self.width

    def _init_display(self):
        # Ensure display surface
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Flappy BPM")

    def _update_speed_from_bpm(self):
        bpm = bpm_data if bpm_data is not None else 0
        if bpm < 50:
            self.bird_speed = 6.5
        elif bpm < 150:
            self.bird_speed = 6.5 - bpm / 42.8
        else:
            self.bird_speed = 3.0

    def step(self, handle_input=False):
        """
        Advance one frame. Returns False if game over, True otherwise.
        handle_input: if True, reads pygame keypresses.
        """
        # Ensure display is initialized
        self._init_display()
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        # Handle input if requested
        keys = pygame.key.get_pressed()
        if handle_input:
            if keys[pygame.K_UP]:
                self.bird_rect.y -= self.bird_speed
            if keys[pygame.K_DOWN]:
                self.bird_rect.y += self.bird_speed
        # Update speed based on BPM
        self._update_speed_from_bpm()
        # Move pipe
        self.pipe_x -= self.pipe_speed
        if self.pipe_x < -self.pipe_width:
            self.score += 1
            self._reset_pipe()
        # Collision detection
        top_pipe = pygame.Rect(self.pipe_x, 0, self.pipe_width, self.pipe_height)
        bottom_pipe = pygame.Rect(
            self.pipe_x,
            self.pipe_height + self.pipe_gap,
            self.pipe_width,
            self.height - self.pipe_height - self.pipe_gap
        )
        if (self.bird_rect.colliderect(top_pipe) or
            self.bird_rect.colliderect(bottom_pipe) or
            self.bird_rect.top <= 0 or
            self.bird_rect.bottom >= self.height):
            return False
        # Draw frame
        self.screen.fill((135, 206, 235))
        pygame.draw.rect(self.screen, (255, 255, 0), self.bird_rect)
        pygame.draw.rect(self.screen, (34, 139, 34), top_pipe)
        pygame.draw.rect(self.screen, (34, 139, 34), bottom_pipe)
        # Ensure font module initialized before rendering text
        if not pygame.font.get_init():
            pygame.font.init()
        # Render text
        font = pygame.font.SysFont(None, 48)
        score_text = font.render(str(self.score), True, (255, 255, 255))
        bpm_text = font.render(str(bpm_data), True, (255, 0, 0))
        self.screen.blit(bpm_text, (10, 10))
        self.screen.blit(score_text, ((self.width - score_text.get_width()) // 2, 20))
        # Update display and tick
        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.fps)
        return True

    def stop(self):
        """Clean up display and stop game"""
        if pygame.display.get_init():
            pygame.display.quit()
        self.screen = None
        self.running = False

    def run(self):
        """Blocking loop if run standalone"""
        self.running = True
        time.sleep(1)
        while self.running:
            if not self.step(handle_input=True):
                self.running = False
        pygame.quit()

__all__ = ['PygameGame']
