# -*- coding: utf-8 -*-
import logging
import os
import re
import sys
import time

class SpringColorFormatter(logging.Formatter):
    """
    Spring Boot style colorized logging formatter with visual multi-task grouping.
    """
    # ANSI escape codes
    RESET = "\033[0m"
    GRAY = "\033[90m"
    BOLD_WHITE = "\033[1;37m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    
    LEVEL_COLORS = {
        logging.DEBUG: "\033[34m",     # Blue
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m" # Bold Red
    }

    TASK_COLORS = [
        "\033[1;36m",  # Bold Cyan
        "\033[1;32m",  # Bold Green
        "\033[1;33m",  # Bold Yellow
        "\033[1;35m",  # Bold Magenta
        "\033[1;34m",  # Bold Blue
        "\033[1;31m",  # Bold Red
    ]

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.pid = os.getpid()
        # Enable ANSI codes on Windows Console
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # Enable virtual terminal processing: 0x0004 | 0x0001 (processed output)
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        msecs = int(record.msecs)
        return f"{t}.{msecs:03d}"

    def format(self, record):
        # 1. Format time (gray)
        formatted_time = f"{self.GRAY}{self.formatTime(record)}{self.RESET}"
        
        # 2. Format level (colorized, padded to 5 chars)
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        level_name = record.levelname
        formatted_level = f"{color}{level_name:>5}{self.RESET}"
        
        # 3. PID (magenta)
        formatted_pid = f"{self.MAGENTA}{self.pid}{self.RESET}"
        
        # 4. Message & Task Extracting
        message = record.getMessage()
        
        # Extract Task Label: e.g. "[Task-1]"
        task_match = re.match(r"^(\[Task-(\d+)\])\s*(.*)$", message)
        if task_match:
            task_label = task_match.group(1)
            task_num = int(task_match.group(2))
            raw_message = task_match.group(3)
            # Cycle through custom task colors
            task_color = self.TASK_COLORS[(task_num - 1) % len(self.TASK_COLORS)]
            formatted_task = f"{task_color}{task_label:<10}{self.RESET}"
        else:
            # Check if there is a general bracket task like [Task-X]
            gen_task_match = re.match(r"^(\[Task-[a-zA-Z0-9_\-]+\])\s*(.*)$", message)
            if gen_task_match:
                task_label = gen_task_match.group(1)
                raw_message = gen_task_match.group(2)
                formatted_task = f"{self.CYAN}{task_label:<10}{self.RESET}"
            else:
                formatted_task = f"{self.GRAY}{'---':<10}{self.RESET}"
                raw_message = message
            
        # 5. Logger name (cyan, shortened format)
        logger_name = record.name
        parts = logger_name.split('.')
        if len(parts) > 1:
            logger_name = f"{parts[0][0]}.{parts[-1]}"
        formatted_logger = f"{self.CYAN}{logger_name:<20}{self.RESET}"
        
        # 6. Status icon color enhancement
        if "✅" in raw_message or "Passed" in raw_message or "成功" in raw_message:
            raw_message = raw_message.replace("✅", f"\033[1;32m✅\033[0m")
        if "❌" in raw_message or "Failed" in raw_message or "失败" in raw_message:
            raw_message = raw_message.replace("❌", f"\033[1;31m❌\033[0m")
            
        # Formulate Spring Boot layout
        spring_log = (
            f"{formatted_time} {formatted_level} {formatted_pid} "
            f"{self.GRAY}---{self.RESET} {formatted_task} "
            f"{formatted_logger} : {raw_message}"
        )
        
        # Append exceptions if any
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            # Colorize traceback in dim red for better error readability
            spring_log = spring_log.rstrip() + "\n" + f"\033[31m{record.exc_text}\033[0m"
            
        return spring_log

def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    Configures the root logger to output Spring-style colorized logs to stdout
    and clean plain-text logs to the specified file (if any).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        
    # 1. Console Stream Handler (Colorized Spring Style)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(SpringColorFormatter())
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # 2. File Handler (Plain-Text logging)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Prevent propagation to double-logging if root is configured
    root_logger.propagate = False
