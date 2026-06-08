import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay

# --- SETUP DIRECTORY ---
save_dir = r"D:\VS codes\Internship(7thSem)\Task-6"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

# --- 1. LOAD AND PREPROCESS DATA ---
print("Loading and preprocessing water quality data...")
df = pd.read_csv('water_potability.csv')

# Handle missing values (this dataset has blanks in pH and Sulfate)
# We fill them with the median so we don't lose half our dataset
df.fillna(df.median(), inplace=True)

# Separate features and target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features (CRITICAL for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data normalized and ready!\n" + "-"*50)

# --- 2 & 3. EXPERIMENT WITH DIFFERENT K VALUES ---
print("\nTraining KNN and experimenting with different K values...")
k_values = range(1, 31)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot K-Value Experiment
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='blue', markerfacecolor='red', markersize=6)
plt.title('Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Testing Accuracy')
plt.grid(True)

k_path = os.path.join(save_dir, "1_K_Value_Experiment.png")
plt.savefig(k_path)
print(f"📈 GRAPH 1 SAVED TO: {k_path}")
print("""--- GRAPH 1 EXPLANATION: CHOOSING K ---
This line chart shows how the accuracy of our model changes as we increase K 
(the number of neighbors it looks at). 
- If K is too small (e.g., K=1), the model is highly sensitive to noise and overfits.
- If K is too large, the model becomes too generalized and underfits.
We look for the peak in this chart to select our optimal K value.""")
plt.show()
plt.close()

# --- 4. EVALUATE MODEL (BEST K) ---
best_k = k_values[np.argmax(accuracies)]
print(f"\nOptimal K found: {best_k}")
print(f"Training final model with K={best_k}...")

# Train final model
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
y_pred_final = final_knn.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"Final Model Accuracy: {final_accuracy * 100:.2f}%")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Unsafe (0)', 'Safe (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (K={best_k})")

cm_path = os.path.join(save_dir, "2_Confusion_Matrix.png")
plt.savefig(cm_path)
print(f"\n📊 GRAPH 2 SAVED TO: {cm_path}")
print("""--- GRAPH 2 EXPLANATION: CONFUSION MATRIX ---
This grid evaluates exactly where our model is getting confused.
- Top-Left: True Negatives (Correctly identified unsafe water).
- Bottom-Right: True Positives (Correctly identified safe water).
- Top-Right & Bottom-Left: Errors (False Positives and False Negatives).
It provides a deeper look at performance beyond just a flat accuracy percentage.""")
plt.show()
plt.close()

# --- 5. VISUALIZE DECISION BOUNDARIES ---
print("\nVisualizing Decision Boundaries (2D Projection)...")
# To visualize a boundary, we must reduce the dataset to just 2 features.
# We will use 'ph' and 'Solids' for this visual demonstration.
X_2d = X_train[['ph', 'Solids']]
scaler_2d = StandardScaler()
X_2d_scaled = scaler_2d.fit_transform(X_2d)

knn_2d = KNeighborsClassifier(n_neighbors=best_k)
knn_2d.fit(X_2d_scaled, y_train)

plt.figure(figsize=(10, 6))
disp = DecisionBoundaryDisplay.from_estimator(
    knn_2d, X_2d_scaled, response_method="predict",
    alpha=0.5, cmap=plt.cm.coolwarm
)
# Scatter the training points on top
disp.ax_.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y_train, edgecolor="k", cmap=plt.cm.coolwarm, s=20)
plt.title(f"KNN Decision Boundary (pH vs. Solids) K={best_k}")
plt.xlabel("Standardized pH")
plt.ylabel("Standardized Solids (TDS)")

db_path = os.path.join(save_dir, "3_Decision_Boundary.png")
plt.savefig(db_path)
print(f"\n🗺️ GRAPH 3 SAVED TO: {db_path}")
print("""--- GRAPH 3 EXPLANATION: DECISION BOUNDARY ---
This plot physically maps how the KNN algorithm divides the space. 
We reduced the data to just 2 features (pH and Solids) to make it visible. 
The colored regions represent what the model will predict (Safe vs Unsafe) 
based on where a new data point lands on this map. Notice how the boundaries 
adapt based on the clustering of the neighboring data points!""")
plt.show()
plt.close()

print("\n*** ALL TASKS COMPLETED SUCCESSFULLY ***")