import sys
import os
import random

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
)
from PySide6.QtCore import QTimer, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pygame

# Placeholder functions for PSI calculation and BPM reading
def calculate_psi(resistance_kg):
    """
    Convert desired resistance in kg to PSI.
    1 kgf/cm^2 is approximately 14.2233 psi.
    """
    try:
        return resistance_kg * 14.2233
    except Exception:
        return 0.0


def get_bpm():
    """
    Simulate reading BPM from a sensor or function.
    Replace with actual implementation.
    """
    return random.randint(60, 100)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resistance-Piston Controller")
        self.resize(800, 600)

        # Central tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        self.home_tab = QWidget()
        self.plot_tab = QWidget()
        self.game_tab = QWidget()

        self.tab_widget.addTab(self.home_tab, "Home")
        self.tab_widget.addTab(self.plot_tab, "Plot")
        self.tab_widget.addTab(self.game_tab, "Game")

        # Setup content of each tab
        self.setup_home_tab()
        self.setup_plot_tab()
        self.setup_game_tab()

        # Game loop timer and screen handle
        self.game_timer = None
        self.screen = None

        # Track tab changes to start/stop game
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def setup_home_tab(self):
        layout = QVBoxLayout()
        # Resistance input
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Desired Resistance (kg):"))
        self.resistance_input = QLineEdit()
        h1.addWidget(self.resistance_input)
        layout.addLayout(h1)
        # PSI output
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Set PSI (calculated):"))
        self.psi_output = QLineEdit(); self.psi_output.setReadOnly(True)
        h2.addWidget(self.psi_output)
        layout.addLayout(h2)
        # BPM output
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Current BPM:"))
        self.bpm_display = QLineEdit(); self.bpm_display.setReadOnly(True)
        h3.addWidget(self.bpm_display)
        layout.addLayout(h3)
        # Buttons
        btn_layout = QHBoxLayout(); btn_layout.addStretch()
        self.reset_button = QPushButton("Reset"); btn_layout.addWidget(self.reset_button)
        self.close_button = QPushButton("Close"); btn_layout.addWidget(self.close_button)
        layout.addLayout(btn_layout)
        self.home_tab.setLayout(layout)
        # Signals
        self.resistance_input.editingFinished.connect(self.update_psi)
        self.reset_button.clicked.connect(self.reset_home)
        self.close_button.clicked.connect(self.close)
        # BPM timer
        self.home_timer = QTimer(self)
        self.home_timer.timeout.connect(self.update_bpm)
        self.home_timer.start(1000)

    def update_psi(self):
        try:
            kg = float(self.resistance_input.text())
        except ValueError:
            kg = 0.0
        self.psi_output.setText(f"{calculate_psi(kg):.2f}")

    def update_bpm(self):
        self.bpm_display.setText(str(get_bpm()))

    def reset_home(self):
        self.resistance_input.clear(); self.psi_output.clear(); self.bpm_display.clear()

    def setup_plot_tab(self):
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.plot_bpm_display = QLineEdit(); self.plot_bpm_display.setReadOnly(True)
        layout.addWidget(QLabel("Current BPM:")); layout.addWidget(self.plot_bpm_display)
        self.plot_tab.setLayout(layout)
        self.plot_data = []
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(1000)

    def update_plot(self):
        bpm = get_bpm(); self.plot_bpm_display.setText(str(bpm))
        self.plot_data.append(bpm)
        if len(self.plot_data) > 20: self.plot_data.pop(0)
        self.figure.clear(); ax = self.figure.add_subplot(111)
        ax.plot(self.plot_data); ax.set_title("BPM Over Time")
        ax.set_ylabel("BPM"); ax.set_xlabel("Seconds")
        self.canvas.draw()

    def setup_game_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Embedded Pygame: activates when this tab is open."))
        self.game_container = QWidget()
        self.game_container.setAttribute(Qt.WA_NativeWindow)
        self.game_container.setMinimumSize(400, 300)
        layout.addWidget(self.game_container)
        self.game_tab.setLayout(layout)

    def start_game(self):
        # Set SDL window ID for embedding
        win_id = self.game_container.winId()
        os.environ['SDL_WINDOWID'] = str(int(win_id))
        # Force X11 driver only on Linux
        if sys.platform.startswith('linux'):
            os.environ['SDL_VIDEODRIVER'] = 'x11'
        # Initialize Pygame display only
        pygame.display.init()
        size = (self.game_container.width(), self.game_container.height())
        self.screen = pygame.display.set_mode(size)
        # Start a QTimer for the game loop (~30 FPS)
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.step_game)
        self.game_timer.start(33)

    def step_game(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_game()
        self.screen.fill((0, 0, 0))
        center = (self.screen.get_width()//2, self.screen.get_height()//2)
        pygame.draw.circle(self.screen, (255, 0, 0), center, 30)
        pygame.display.flip()

    def stop_game(self):
        if self.game_timer:
            self.game_timer.stop(); self.game_timer = None
        pygame.display.quit(); pygame.quit(); self.screen = None

    def on_tab_changed(self, index):
        if index == 2:
            self.start_game()
        else:
            self.stop_game()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Dark-blue global style
    app.setStyleSheet("""
        QMainWindow, QWidget { background-color: #00008B; color: white; }
        QTabWidget::pane { background-color: #00008B; }
        QTabBar::tab { background: #00008B; color: white; padding: 8px; }
        QTabBar::tab:selected { background: #000060; }
        QLineEdit, QLabel { background-color: #00008B; color: white; border: 1px solid #FFF; }
        QPushButton { background-color: #000060; color: white; border-radius: 4px; padding: 6px 12px; }
        QPushButton:hover { background-color: #000080; }
    """
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec())