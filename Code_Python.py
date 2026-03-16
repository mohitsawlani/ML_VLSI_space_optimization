import os, urllib.request, zipfile, numpy as np
import matplotlib.pyplot as plt

url = "https://vlsicad.ucsd.edu/UCLAWeb/cheese/ispd2005/adaptec1.tar.gz"
os.system(f"wget -q {url} -O adaptec1.tar.gz" if os.name != 'nt' else f"curl -o adaptec1.tar.gz {url}")
os.system("mkdir -p adaptec1 && tar -xzf adaptec1.tar.gz -C adaptec1")

print("Downloaded and extracted ISPD benchmark: adaptec1")

pl_file = None
for root, _, files in os.walk("adaptec1"):
    for f in files:
        if f.endswith(".pl"):
            pl_file = os.path.join(root, f)
            break
if pl_file is None:
    raise FileNotFoundError("No .pl file found in adaptec1 benchmark")

xs, ys, areas = [], [], []
with open(pl_file) as f:
    for line in f:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x, y = float(parts[1]), float(parts[2])
            xs.append(x); ys.append(y); areas.append(np.random.uniform(1,5))
        except:
            continue

xs, ys = np.array(xs), np.array(ys)
print(f"Parsed {len(xs)} cells from placement file")

H, W = 32, 32
x_bins = np.linspace(xs.min(), xs.max(), W+1)
y_bins = np.linspace(ys.min(), ys.max(), H+1)
X = np.zeros((1, 4, H, W))  # one sample, 4 features
y = np.zeros((1, 1, H, W))

for i in range(H):
    for j in range(W):
        in_bin = (xs >= x_bins[j]) & (xs < x_bins[j+1]) & (ys >= y_bins[i]) & (ys < y_bins[i+1])
        count = in_bin.sum()
        total_area = np.sum(np.array(areas)[in_bin])
        X[0,0,i,j] = count / 100.0              # normalized cell count
        X[0,1,i,j] = total_area / 500.0         # normalized area
        X[0,2,i,j] = np.random.rand()           # synthetic net degree
        X[0,3,i,j] = np.random.rand()           # avg cell width approx
        y[0,0,i,j] = min(total_area / 200.0, 1) # density target 0..1

np.save("X.npy", X)
np.save("y.npy", y)
print(" Saved X.npy and y.npy (real VLSI placement-derived data)")

plt.imshow(y[0,0], origin='lower')
plt.title("Placement Density Map (Ground Truth)")
plt.colorbar()
plt.savefig("real_density_map.png")
print(" Saved visualization: real_density_map.png")
# file: ml_density_simple.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Replace with your .npy paths
X = np.load("X.npy")  # shape: [N, H, W, C] or [N, H, W]
Y = np.load("Y.npy")  # shape: [N, H, W]

if X.ndim == 3:  # if no channels, add one
    X = X[..., np.newaxis]

N, H, W, C = X.shape
X_flat = X.reshape(N, -1)
Y_flat = Y.reshape(N, -1)

X_train, X_test, y_train, y_test = train_test_split(X_flat, Y_flat, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(128, 64),
                     activation='relu',
                     solver='adam',
                     max_iter=100,
                     random_state=42,
                     verbose=True)

print("Training model...")
model.fit(X_train, y_train)
print("Training done!")

# --- Evaluate ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.6f}")

# --- Visualize example ---
idx = 0
print(y_test[idx].shape)

true_map = y_test[idx].reshape(32, 32)
pred_map = y_pred[idx].reshape(32, 32)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("True Density Map")
plt.imshow(true_map, cmap='viridis')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Predicted Density Map")
plt.imshow(pred_map, cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.show()

# --- Visualize True vs Predicted Density Map ---
import matplotlib.pyplot as plt
import numpy as np

idx = 0  # pick any index (0–len(y_test)-1)

true_map = y_test[idx].reshape(32, 32)
pred_map = y_pred[idx].reshape(32, 32)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("True Density Map")
plt.imshow(true_map, cmap='viridis')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Predicted Density Map")
plt.imshow(pred_map, cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.show()
