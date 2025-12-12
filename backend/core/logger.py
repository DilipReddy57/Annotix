import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Add FileHandler here if needed
    ]
)

logger = logging.getLogger("autolabel_agent")

def get_logger(name: str):
    return logging.getLogger(f"autolabel_agent.{name}")
