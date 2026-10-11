import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
df = pd.read_csv("out/loss.csv")

# Initialize the plot size
plt.figure(figsize=(9, 5))

# Plot Train Loss and Validation Loss against Step
plt.plot(
    df["step"],
    df["train_loss"],
    marker="o",
    linestyle="-",
    color="blue",
    label="Train Loss",
)
plt.plot(
    df["step"],
    df["val_loss"],
    marker="s",
    linestyle="--",
    color="orange",
    label="Validation Loss",
)

# Add titles, labels, and visual grid
plt.title("Model Training & Validation Loss Curve", fontsize=14)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11)

# Save or show the plot
plt.tight_layout()
plt.show()
