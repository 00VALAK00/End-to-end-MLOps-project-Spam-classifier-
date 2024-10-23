import sys
from pathlib import Path
from zenml.logger import get_logger

parent_dir = Path(__package__).parent
sys.path.append(str(parent_dir))

logger = get_logger(__file__)
