import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------- Step 1: Load Dataset -----------------
file_path = "mobile_addiction_data.csv"  # ✅ Replace with your actual file path
data = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully!")
print("Columns in dataset:\n", data.columns.tolist())

# ----------------- Step 2: Clean Data -----------------
data.dropna(inplace=True)
data = data[data["Daily_Screen_Time_Hours"] <= 24]
data = data[data["Sleep_Hours"] <= 24]

# ----------------- Step 3: Select Numeric Features for Clustering -----------------
numeric_cols = [
    "Daily_Screen_Time_Hours",
    "Phone_Unlocks_Per_Day",
    "Social_Media_Usage_Hours",
    "Gaming_Usage_Hours",
    "Streaming_Usage_Hours",
    "Messaging_Usage_Hours",
    "Work_Related_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Time_Spent_With_Family_Hours"
]

print("\n✅ Numeric features selected for clustering:\n", numeric_cols)

# ----------------- Step 4: Scale Data -----------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numeric_cols])

# ----------------- Step 5: Determine Optimal K -----------------
inertia, sil_scores = [], []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled_data, km.labels_))

# Elbow Plot
plt.figure(figsize=(6, 4))
plt.plot(range(2, 7), inertia, marker="o")
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# ----------------- Step 6: Fit KMeans -----------------
best_k = 4  # ✅ Adjust based on Elbow plot or silhouette analysis
kmeans = KMeans(n_clusters=best_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# ----------------- Step 7: Cluster Summary Stats -----------------
cluster_summary = data.groupby('Cluster')[numeric_cols + ['Mental_Health_Score']].mean().round(2)
cluster_counts = data['Cluster'].value_counts().sort_index()
print("\n📊 Cluster Summary (Mean Values):\n", cluster_summary)
print("\nNumber of users per cluster:\n", cluster_counts)

# ----------------- Step 8: Visualizations -----------------
# Screen Time vs Sleep
plt.figure(figsize=(8,6))
sns.scatterplot(x="Daily_Screen_Time_Hours", y="Sleep_Hours", hue="Cluster", data=data, palette="Set2")
plt.title("Clusters: Screen Time vs Sleep Hours")
plt.show()

# Mental Health Score Distribution
plt.figure(figsize=(8,6))
sns.boxplot(x="Cluster", y="Mental_Health_Score", data=data, palette="Set3")
plt.title("Mental Health Score across Clusters")
plt.show()

# ----------------- Step 9: Automatic Insights -----------------
print("\n💡 Automatic Insights by Cluster:")
for i, row in cluster_summary.iterrows():
    print(f"\nCluster {i} ({cluster_counts[i]} users):")
    
    # Digital Burnout
    if row["Daily_Screen_Time_Hours"] > 6 and row["Sleep_Hours"] < 6:
        print("  • Digital Burnout: High screen time + low sleep.")
    
    # Balanced Lifestyle
    elif row["Daily_Screen_Time_Hours"] <= 5 and row["Sleep_Hours"] >= 7:
        print("  • Balanced Lifestyle: Moderate screen time + good sleep.")
    
    # Work-Oriented Users
    if row.get("Work_Related_Usage_Hours",0) > 5:
        print("  • Work-Oriented Users: High work-related usage.")
    
    # Sedentary Social Media Addicts
    if row.get("Social_Media_Usage_Hours",0) > 4 and row.get("Physical_Activity_Hours",0) < 1:
        print("  • Sedentary Social Media Addicts: High social media + low activity.")
    
    # Mental Health Alert
    if row.get("Mental_Health_Score",100) < 50:
        print("  • Mental Health Concern: Lower mental wellbeing detected.")

# ----------------- Step 10: Regression to Predict Mental Health Score -----------------
target = "Mental_Health_Score"
features = numeric_cols.copy()  # ✅ Use behavioral numeric features only

X = data[features]
y = data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Regression Model Evaluation:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

# Feature Importance
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop Feature Importances:")
print(importances)

# Feature Importance Plot
plt.figure(figsize=(8,6))
importances[:10].plot(kind='barh')
plt.title("Top 10 Features for Predicting Mental Health Score")
plt.gca().invert_yaxis()
plt.show()

# ----------------- Step 11: Export for Tableau -----------------
data['Predicted_Mental_Health_Score'] = rf.predict(X)

output_path = "/Users/hemapriyadharshni/Documents/Data Mining DA/processed_smartphone_usage.csv"
data.to_csv(output_path, index=False)

print(f"\n✅ CSV saved successfully for Tableau at:\n{output_path}")
print("👉 Columns available for Tableau Dashboard:")
print(data.columns.tolist())