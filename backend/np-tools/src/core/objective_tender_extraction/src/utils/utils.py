import pathlib
import logging

def init_logger(name: str, path_logs: pathlib.Path) -> logging.Logger:
    """
    Initialize a logger with a specified name and log file path.

    Parameters
    ----------
    name : str
        The name of the logger.
    path_logs : Path
        The directory path where the log file will be stored.

    Returns
    -------
    logging.Logger
        The initialized logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create path_logs dir if it does not exist
    path_logs.mkdir(parents=True, exist_ok=True)

    # Create handlers
    file_handler = logging.FileHandler(path_logs / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to the handlers
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)

    # Add the handlers to the logger if they are not already added
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger