import os
import sys

# Add the parent directory to the Python path
# This allows test files to import from the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# You can add any common test utilities or fixtures here
def setup_test_environment():
    """Set up any environment variables or configurations needed for tests."""
    os.environ['TESTING'] = 'True'

# Run the setup function when the tests package is imported
setup_test_environment()