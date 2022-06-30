import os

directory = ("data/")
files_in_directory = os.listdir(directory)
filtered_files = [file for file in files_in_directory if file != "taxy_trips_ny_2016-06-01to03_3%sample.csv"]

for file in filtered_files:
    path_to_file = os.path.join(directory, file)
    os.remove(path_to_file)


hello_world_path = "../hello_world_examples"
for file in os.listdir(hello_world_path):
    if not file.endswith(".py") and not file.endswith(".jpg"):
        if not os.path.isdir(os.path.join(hello_world_path, file)):
            os.remove(os.path.join(hello_world_path, file))