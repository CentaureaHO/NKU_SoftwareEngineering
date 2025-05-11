import os

def save_file(file, folder='./uploads'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, file.filename)
    file.save(path)
    return path
