import os
import shutil

# Default Windows Hugging Face cache path
cache_path = os.path.expanduser(r"~\.cache\huggingface\hub\models--CIDAS--clipseg-rd16")

if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
    print("Corrupted cache successfully deleted!")
else:
    print("Folder already deleted or not found.")