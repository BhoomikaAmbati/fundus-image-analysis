import numpy as np
import os
import glob
import csv

pca_results_path = os.path.join("results", "pca_results")

folder_path = "pca_result_activation_layer"
folder_path = os.path.join(pca_results_path, folder_path)
output = "significance_activation_17_d4_more_than_99.csv"
output = os.path.join(pca_results_path, output)

# Get all subdirectories inside the main folder
subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

csv_files = []
img_names = []

# Loop through each subdirectory and find the CSV file
for subdir in subdirs:
    img_names.append(subdir)
    csv_path = os.path.join(folder_path, subdir, f"{subdir}_ev.csv")
    if os.path.exists(csv_path):
        # Replace backslashes with forward slashes
        csv_files.append(csv_path.replace("\\", "/"))

# Print the list of CSV files
print("Found ", len(csv_files), " csv files")

with open(output, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["S/n","ImageName", "L", "Information retained", "Loss in Information"])
    # Specify the same file path
    for i in range(len(csv_files)):
        img = csv_files[i]
        file_path = f"{img}"
        # Load from CSV
        ev = np.loadtxt(file_path, delimiter=",")
        ev = ev[1: -1]
        l = 1
        sig = 0
        selected = ev[:l]
        signif = np.sum(selected)/np.sum(ev)
        while(sig < 0.99):
            selected = ev[:l]
            sig = np.sum(selected)/np.sum(ev)
            l = l + 1
        writer.writerow([i+1, img_names[i], l-1, sig, (1-sig)])
print(f"Successfully stored results in {output}")