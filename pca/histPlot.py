import pandas as pd
import matplotlib.pyplot as plt
import io, os

significance_path = os.path.join("results", "pca_results")
output_path = os.path.join(significance_path, "hist_plot_sig_99_per.png")
def save_image(image, save_path):
    with open(save_path, "wb") as f:
        f.write(image.getbuffer())

data = os.path.join(significance_path, "significance_activation_17_d4_more_than_99.csv")
df = pd.read_csv(data)
column = 'L'

plt.figure(figsize=(12, 8))
plt.hist(df[column],color="blue", edgecolor="black", alpha=0.7)

# Labels and title
plt.xlabel(f"Number of Selected images (L)", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Show the plot
img_buf = io.BytesIO()
plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
plt.close()
img_buf.seek(0)
save_image(img_buf, output_path)