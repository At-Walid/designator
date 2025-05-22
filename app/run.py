import sys
import os
import threading
import time
import subprocess
import importlib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QComboBox, QLineEdit, QGroupBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QFileSystemWatcher
import win32gui
import win32con
import atexit
import psutil
import shutil
SEARCH_ALGORITHMS = {
    "DESIGNATE": "search2_feat_mars",
    "DESIGNATE_SINGLE": "search1_mars",
    "DESIGNATE_PIXEL": "search2_pixacc_mars",
    "DESIGNATE_NoGAN": "nogan_mars",
    "Random": "search_random_mars",
}

class SimulatorGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🚀 Mars Simulator Controller")
        self.setGeometry(100, 100, 1280, 900)
        self.simulator_path = "app/MarsSim/WindowsNoEditor/Mars.exe"
        self.simulator_process = None
        self.simulator_hwnd = None
        self.search_running = False
        self.archive_path = "./results/archive.txt"
        
        self.current_search_module = None
        self.search_module_name = ""

        # Set global font style
        self.setFont(QFont("Arial", 12))

        # Main Layout
        main_layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Mars Simulator Controller")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        main_layout.addWidget(title_label)

        # Search Type Dropdown
        dropdown_layout = QHBoxLayout()
        self.search_dropdown = QComboBox()
        self.search_dropdown.addItems(SEARCH_ALGORITHMS.keys())
        self.search_dropdown.currentIndexChanged.connect(self.load_search_module)
        dropdown_layout.addWidget(QLabel("🔍 Search Type:"))
        dropdown_layout.addWidget(self.search_dropdown)
        main_layout.addLayout(dropdown_layout)

        # Search Parameters
        params_layout = QHBoxLayout()
        self.pop_size_input = QLineEdit("12")
        self.n_gen_input = QLineEdit("100")
        params_layout.addWidget(QLabel("👥 Population Size:"))
        params_layout.addWidget(self.pop_size_input)
        params_layout.addWidget(QLabel("🔄 Generations:"))
        params_layout.addWidget(self.n_gen_input)
        main_layout.addLayout(params_layout)

        # Buttons Layout
        button_layout = QHBoxLayout()

        self.start_sim_button = QPushButton("▶️ Start Simulator")
        self.start_sim_button.clicked.connect(self.start_simulator)
        self.style_button(self.start_sim_button)
        button_layout.addWidget(self.start_sim_button)

        self.stop_sim_button = QPushButton("⏹ Stop Simulator")
        self.stop_sim_button.clicked.connect(self.stop_simulator)
        self.style_button(self.stop_sim_button)
        button_layout.addWidget(self.stop_sim_button)

        self.start_search_button = QPushButton("🔍 Begin Search")
        self.start_search_button.setEnabled(False)
        self.start_search_button.clicked.connect(self.start_search)
        self.style_button(self.start_search_button)
        button_layout.addWidget(self.start_search_button)

        main_layout.addLayout(button_layout)

        # Simulator Section
        sim_group = QGroupBox("🛰 Simulation Window")
        sim_layout = QVBoxLayout()
        self.simulator_widget = QFrame(self)
        self.simulator_widget.setFrameShape(QFrame.Box)
        self.simulator_widget.setFixedSize(700, 450) 
        sim_layout.addWidget(self.simulator_widget)
        sim_group.setLayout(sim_layout)
        main_layout.addWidget(sim_group)

        # Progress Bar
        # self.progress_bar = QProgressBar()
        # self.progress_bar.setMaximum(100)
        # main_layout.addWidget(QLabel("⏳ Search Progress"))
        # main_layout.addWidget(self.progress_bar)

        # Image display section
        image_group = QGroupBox("📷 Image Outputs")
        image_layout = QHBoxLayout()

        self.label_image_label = QLabel()
        self.label_image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.create_image_widget("📝 Label Image", self.label_image_label))

        self.simulated_image_label = QLabel()
        self.simulated_image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.create_image_widget("📸 Simulated Image", self.simulated_image_label))

        self.realistic_image_label = QLabel()
        self.realistic_image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.create_image_widget("🌍 Realistic Image", self.realistic_image_label))

        self.prediction_image_label = QLabel()
        self.prediction_image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.create_image_widget("🔮 Prediction Image", self.prediction_image_label))


        image_group.setLayout(image_layout)
        main_layout.addWidget(image_group)

        # Fitness Score Labels
        self.fitness_label = QLabel("🏆 Fitness Scores: N/A")
        main_layout.addWidget(self.fitness_label)

        # Archive Size Display
        self.archive_size_label = QLabel("📂 Archive Size: 0")
        main_layout.addWidget(self.archive_size_label)

        self.setLayout(main_layout)

        # Monitor archive.txt for real-time updates
        self.file_watcher = QFileSystemWatcher()
        if os.path.exists(self.archive_path):
            self.file_watcher.addPath(self.archive_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.watch_archive)
        self.timer.start(2000)

        self.load_search_module()
        self.watch_archive()
        self.simulator_process = None
        self.simulator_path = "app/MarsSim/WindowsNoEditor/Mars.exe"
        atexit.register(self.stop_simulator)

    def style_button(self, button):
        """Applies styling to buttons."""
        button.setFont(QFont("Arial", 12, QFont.Bold))
        button.setStyleSheet("background-color: #007BFF; color: white; padding: 10px; border-radius: 5px;")

    def create_image_widget(self, title, label_widget):
        """Creates a widget containing an image and its label."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(title, alignment=Qt.AlignCenter))
        layout.addWidget(label_widget)
        widget.setLayout(layout)
        return widget

    def load_search_module(self):
        """Loads the selected search module dynamically and updates the archive path."""
        selected_search = self.search_dropdown.currentText()

        if selected_search not in SEARCH_ALGORITHMS:
            print(f"❌ Error: Unknown search type '{selected_search}'")
            return

        search_module_name = SEARCH_ALGORITHMS[selected_search]

        try:
            self.search_module_name = search_module_name
            self.current_search_module = importlib.import_module(search_module_name)
            self.gagan = self.current_search_module.GaGan()
        except ImportError as e:
            print(f"❌ Failed to import module {search_module_name}: {e}")
            return

        # 🔹 Update archive path dynamically
        self.archive_path = f"./results/{selected_search}/archive.txt"
        print(f"📂 Archive path set to: {self.archive_path}")

        # Re-add the new archive path to the file watcher
        self.file_watcher.removePaths(self.file_watcher.files())  # Clear old watch paths
        if os.path.exists(self.archive_path):
            self.file_watcher.addPath(self.archive_path)

        # Ensure the UI updates with the latest archive data
        self.watch_archive()


    def set_windowed_mode(self):
        """Forces the Airsim window to run in windowed mode before embedding."""
        if self.simulator_hwnd:
            style = win32gui.GetWindowLong(self.simulator_hwnd, win32con.GWL_STYLE)
            style &= ~win32con.WS_POPUP  # Remove fullscreen mode
            style |= win32con.WS_OVERLAPPEDWINDOW  # Set as a normal window
            win32gui.SetWindowLong(self.simulator_hwnd, win32con.GWL_STYLE, style)

            # Resize and reposition the window
            time.sleep(10)
            win32gui.SetWindowPos(
                self.simulator_hwnd, None,
                100, 100, 800, 600,  # Adjust position and window size as needed
                win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED
            )
            print("✅ Simulator set to windowed mode.")
        else:
            print("❌ Simulator window handle not found.")

    def start_simulator(self):
        """Starts the Mars simulator and ensures it runs in windowed mode before embedding."""
        custom_settings = "app/settings.json"
        airsim_settings_dir = os.path.join(os.environ["USERPROFILE"], "Documents", "AirSim")
        airsim_settings = os.path.join(airsim_settings_dir, "settings.json")

        # Ensure directory exists
        os.makedirs(airsim_settings_dir, exist_ok=True)

        # Copy custom settings
        shutil.copy(custom_settings, airsim_settings)
        if self.simulator_process is None:
            self.simulator_process = subprocess.Popen(
                [self.simulator_path, "--settings", "app/settings.json"],  # 🔹 Specify custom settings
                creationflags=subprocess.CREATE_NO_WINDOW  # Prevents fullscreen launch
            )

            time.sleep(10)  # Give time for the simulator to initialize
            # self.gagan.begin_server("app/MarsSim/WindowsNoEditor/Mars.exe", 'PhysXCar')

            # Ensure the window is found
            self.simulator_hwnd = None
            for _ in range(10):  # Retry multiple times in case the window appears late
                self.simulator_hwnd = self.find_window_handle("Mars")
                if self.simulator_hwnd:
                    break
                time.sleep(1)

            if self.simulator_hwnd:
                self.set_windowed_mode()  # 🔹 Make sure this method is now available
                self.embed_simulator_window()
                self.start_search_button.setEnabled(True)
            else:
                print("❌ Error: Unable to find the simulator window!")



    def stop_simulator(self):
        """Stops the Mars simulator process if running."""
        if self.simulator_process:
            print("⏹ Stopping Mars Simulator...")
            self.simulator_process.terminate()
            try:
                self.simulator_process.wait(timeout=5)  # Wait for the process to exit
            except subprocess.TimeoutExpired:
                self.simulator_process.kill()  # Force kill if it hangs
            self.simulator_process = None
            print("✅ Simulator stopped.")

        # Ensure any lingering processes are killed
        self.kill_process_by_name("Mars.exe")

    def kill_process_by_name(self, process_name):
        """Kills any lingering Mars.exe processes."""
        for proc in psutil.process_iter(attrs=['pid', 'name']):
            if proc.info['name'].lower() == process_name.lower():
                print(f"🔴 Killing lingering process: {proc.info['name']} (PID: {proc.info['pid']})")
                proc.kill()

    def closeEvent(self, event):
        """Handle application close event."""
        self.stop_simulator()
        event.accept()

    def start_search(self):
        """Starts the search process in a separate thread."""
        if not self.search_running:
            self.search_running = True
            threading.Thread(target=self.run_search, daemon=True).start()

    def run_search(self):
        """Runs the search algorithm and updates images dynamically."""
        print(self.search_module_name)
        if not 'random' in self.search_module_name:
            self.gagan.searchAlgo(100, 12, 1)
        else:
            self.gagan.searchAlgo(1)
       
    def watch_archive(self):
        """Continuously monitors archive.txt for new entries and updates images and fitness scores."""
        if os.path.exists(self.archive_path):
            with open(self.archive_path, "r") as file:
                lines = file.readlines()

            self.archive_size_label.setText(f"📂 Archive Size: {len(lines)}")  # Update archive size

            if not lines:
                return  # No data to process yet

            try:
                latest_entry = eval(lines[-1])  # Convert string to dictionary
            except:
                print("⚠️ Error parsing archive entry")
                return

            # Extract image paths from selected search folder
            search_type = self.search_dropdown.currentText()  # Get current search type
            results_dir = f"./results/{search_type}"  # Dynamically update path

            real_img = os.path.join(results_dir, os.path.basename(latest_entry['img']))

            x, y = latest_entry['individual']
            x_str = str(x)  
            y_str = str(y)

            label_img = f"{results_dir}/L_{x_str}_{y_str}.png"
            simulated_img = f"{results_dir}/S_{x_str}_{y_str}.png"
            prediction_img = f"{results_dir}/P_{x_str}_{y_str}.png"
            # Debugging logs
            # print(f"📷 Checking images in: {results_dir}")
            # print(f" - Real Image: {real_img}")
            # print(f" - Label Image: {label_img}")
            # print(f" - Simulated Image: {simulated_img}")

            # Verify image existence
            # print(real_img, label_img, simulated_img)
            missing_files = [p for p in [real_img, label_img, simulated_img] if not os.path.exists(p)]
            # if missing_files:
            #     print(f"❌ Missing images: {missing_files}")
            # else:
            #     print("✅ All images exist.")

            # Wait for images to be available before updating UI
            if not self.wait_for_images([real_img, label_img, simulated_img]):
                print("❌ Images did not appear in time!")
                return

            # Update GUI with new images
            # print("✅ Updating GUI with new images.")
            # self.update_fitness_display(*latest_entry['F'])
            self.display_image(label_img, self.label_image_label)
            self.display_image(simulated_img, self.simulated_image_label)
            self.display_image(real_img, self.realistic_image_label)
            self.display_image(prediction_img, self.prediction_image_label)
            self.fitness_label.setText(f"🏆 Fitness Scores: {latest_entry['F'][0]:.4f}, {latest_entry['F'][1]:.4f}")


    def find_window_handle(self, window_name):
        """Finds the window handle (HWND) of the running simulator."""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd) and window_name in win32gui.GetWindowText(hwnd):
                extra.append(hwnd)

        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        return hwnds[0] if hwnds else None

    def embed_simulator_window(self):
        """Embeds the simulator window inside the PyQt application with a smaller size."""
        if self.simulator_hwnd:
            win32gui.SetParent(self.simulator_hwnd, self.simulator_widget.winId())  # Attach to simulator widget

            # Adjust window style to remove title bar
            style = win32gui.GetWindowLong(self.simulator_hwnd, win32con.GWL_STYLE)
            win32gui.SetWindowLong(self.simulator_hwnd, win32con.GWL_STYLE, style & ~win32con.WS_CAPTION)

            # Ensure proper window size and positioning
            sim_width = self.simulator_widget.width() - 20  # Small padding to avoid overflow
            sim_height = int(self.simulator_widget.height() * 0.85)  # Reduce height to fit properly
            sim_x = self.simulator_widget.x()
            sim_y = self.simulator_widget.y()

            win32gui.SetWindowPos(
                self.simulator_hwnd, None,
                sim_x, sim_y,
                sim_width, sim_height,
                win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE | win32con.SWP_FRAMECHANGED
            )


    def wait_for_images(self, paths, timeout=5):
        """Waits for images to be available before displaying them."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if all(os.path.exists(p) for p in paths):
                return True
            time.sleep(0.5)  # Wait a bit before checking again
        return False

    def update_fitness_display(self, fitness1, fitness2):
        """Displays the fitness scores on the GUI."""
        print(f"🏆 Fitness Scores: {fitness1:.4f}, {fitness2:.4f}")

    def display_image(self, img_path, label_widget):
        """Displays the image and forces the UI to refresh."""
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            pixmap = pixmap.scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_widget.setPixmap(pixmap)
            label_widget.setText("")  # Clear text
            QApplication.processEvents()  # Force UI refresh
        else:
            print(f"⚠️ Image file not found: {img_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimulatorGUI()
    window.show()
    sys.exit(app.exec_())
