from loguru import logger

from src.config import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)

logger.remove()

logger.add(
    LOG_DIR / "application.log",
    rotation="10 MB",
    level="INFO"
)

logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO"
)