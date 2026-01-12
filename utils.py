from datetime import datetime
from colorama import Fore, Style, init
import re

# Initialize colorama for Windows
init(autoreset=True)

log_file = "anything_else_log.txt"

# ANSI escape stripper (for log file)
ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def strip_colors(text):
    return ANSI_ESCAPE.sub('', text)

def initialize(filename):
    global log_file
    log_file = filename

def prints(message, level=None):
    """
    level: None | 'info' | 'warning' | 'error'
    """

    # Color mapping
    level_colors = {
        "info": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED + Style.BRIGHT,
    }

    # Apply color ONLY if level is provided and valid
    if level in level_colors:
        console_message = f"{level_colors[level]}{message}"
    else:
        console_message = message  # plain, no color

    # Print to console
    print(console_message)

    # Prepare clean message for log
    clean_message = strip_colors(message)

    # Timestamp for log file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Include level in log only if present
    if level:
        full_message = f"[{timestamp}] [{level.upper()}] {clean_message}"
    else:
        full_message = f"[{timestamp}] {clean_message}"

    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")
