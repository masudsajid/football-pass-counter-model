# main.py
"""
Main entry point for the Football Pass Counter application.
Run this file to start the Streamlit web interface.
"""

from ui import StreamlitApp


def main():
    """Main entry point for the application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
