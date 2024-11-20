import os

def list_files(directory):
    file_arr = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_arr.append(filename)
    return file_arr

layout_files = list_files('overcookedgym/overcooked_ai/src/overcooked_ai_py/data/layouts')

LAYOUT_LIST = [os.path.splitext(filename)[0] for filename in layout_files]
