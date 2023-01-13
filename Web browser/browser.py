import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView


class Browser(QWebEngineView):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt Web Browser")

        # Create the navigation buttons
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.back)
        self.forward_button = QPushButton("Forward")
        self.forward_button.clicked.connect(self.forward)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)

        # Create a layout for the navigation buttons
        navigation_layout = QHBoxLayout()
        navigation_layout.addWidget(self.back_button)
        navigation_layout.addWidget(self.forward_button)
        navigation_layout.addWidget(self.refresh_button)

        # Set the layout for the browser window
        layout = QHBoxLayout()
        layout.addLayout(navigation_layout)
        layout.addWidget(self)
        self.setLayout(layout)

        # Load a default URL
        self.load(QUrl("http://www.google.com"))

    def back(self):
        """Go back to the previous page"""
        self.page().triggerAction(QWebEnginePage.Back)

    def forward(self):
        """Go forward to the next page"""
        self.page().triggerAction(QWebEnginePage.Forward)

    def refresh(self):
        """Refresh the current page"""
        self.page().triggerAction(QWebEnginePage.Reload)

app = QApplication(sys.argv)
browser = Browser()
browser.show()
sys.exit(app.exec_())
