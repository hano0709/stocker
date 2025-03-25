# Clean up existing files
import os
import shutil

# Force cleanup of existing files
if os.path.exists('data'):
    shutil.rmtree('data')
if os.path.exists('models'):
    shutil.rmtree('models')
if os.path.exists('config.json'):
    os.remove('config.json')

print("Cleaned up existing files")