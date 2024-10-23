import sys
from pathlib import Path

# Set the main project directory to the Python path
main_dir = Path(__package__).parent
sys.path.append(str(main_dir))
