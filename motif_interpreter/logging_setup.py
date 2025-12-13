"""
Logging configuration
"""
import logging

def setup_logging(level=logging.DEBUG):
    """Setup logging configuration"""
    logger = logging.getLogger("motif_interpreter")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

# Create a global logger instance
logger = setup_logging()