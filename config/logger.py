# config/logger.py
import logging

class ColoredFormatter(logging.Formatter):
    """
    A custom formatter class that adds color codes to the log output.
    """
    # Setting up color codes
    COLOR_CODES = {
        "reset"  : "\033[0m",
        "cyan"   : "\033[36m",
        "green"  : "\033[32m", 
        "yellow" : "\033[33m",
        "red"    : "\033[31m",
        "magenta": "\033[35m",
        "blue"   : "\033[34m",
    }
    
    PARTS_FORMAT = {
        "asctime"  : COLOR_CODES["blue"]    + "%(asctime)s"    + COLOR_CODES["reset"],
        "levelname": COLOR_CODES["red"]     + "%(levelname)-5s"+ COLOR_CODES["reset"], 
        "filename" : COLOR_CODES["green"]   + "%(filename)-20s"+ COLOR_CODES["reset"],
        "funcName" : COLOR_CODES["magenta"] + "%(funcName)-20s"+ COLOR_CODES["reset"],
        "lineno"   : COLOR_CODES["yellow"]  + "%(lineno)4d"    + COLOR_CODES["reset"],
        "message"  : COLOR_CODES["reset"]   + "%(message)s"    + COLOR_CODES["reset"]
    }

    def format(self, record):
        """
        Overrides the default format method to apply color codes to each part of the log record.
        
        Args:
            record (logging.LogRecord): The log record to be formatted.
        
        Returns:
            str: The formatted log record with color codes applied.
        """
        format_orig = self._fmt
        
        # Apply color codes to each part of the log record  
        self._style._fmt = "  -  ".join(self.PARTS_FORMAT[part] for part in self.PARTS_FORMAT)
        
        # Call the original formatter to do the sprintf-style formatting
        result = logging.Formatter.format(self, record)
        
        # Restore the original format configured by the user
        self._fmt = format_orig
        
        return result

def setup_logging(debug=False):
    """
    Sets up the logging configuration with a colored formatter and a console handler.
    
    Args:
        debug (bool, optional): If True, sets the log level to DEBUG. Defaults to False.
    """
    # Set up colored logging
    colored_formatter = ColoredFormatter(
        "%(asctime)s  -  %(levelname)s  -  %(filename)s  -  %(funcName)s:%(lineno)d  -  %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.addHandler(console_handler)

    # Set the logging level for the 'paramiko' module to WARNING 
    logging.getLogger("paramiko").setLevel(logging.WARNING)