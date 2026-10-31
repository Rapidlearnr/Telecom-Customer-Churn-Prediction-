
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# 1. Load the Dataset



df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
print("\nFirst Five Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# 2. Data Cleaning

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.drop_duplicates()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# 3. Exploratory Data Analysis


# Graph 1: Customer Churn Distribution

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

churn_percentage = df["Churn"].value_counts(normalize=True) * 100

print("\nChurn Percentage:")
print(churn_percentage.round(2))

plt.figure(figsize=(8, 5))

ax = df["Churn"].value_counts().reindex(["No", "Yes"]).plot(
    kind="bar"
)

plt.title(
    "Customer Churn Distribution",
    fontsize=16,
    fontweight="bold",
    pad=15
)
plt.xlabel("Customer Churn Status")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.25)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

plt.tight_layout()
plt.show()


# Graph 2: Customer Churn by Contract Type

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Contract Type:")
print(contract_churn.round(2))

ax = contract_churn.reindex(columns=["No", "Yes"]).plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title(
    "Customer Churn by Contract Type",
    fontsize=16,
    fontweight="bold",
    pad=15
)
plt.xlabel("Contract Type")
plt.ylabel("Customers (%)")
plt.xticks(rotation=0)
plt.legend(
    ["Retained (No Churn)", "Churned"],
    title="Customer Status"
)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# Graph 3: Customer Churn Rate by Tenure

tenure_churn = df.groupby("tenure")["Churn"].apply(
    lambda values: (values == "Yes").mean() * 100
)

plt.figure(figsize=(10, 5))

plt.plot(
    tenure_churn.index,
    tenure_churn.values,
    marker=".",
    linewidth=2
)

plt.title(
    "Customer Churn Rate by Tenure",
    fontsize=16,
    fontweight="bold",
    pad=15
)
plt.xlabel("Customer Tenure (Months)")
plt.ylabel("Churn Rate (%)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Graph 4: Monthly Charges: Churned vs Retained Customers

plt.figure(figsize=(8, 5))

df.boxplot(
    column="MonthlyCharges",
    by="Churn",
    grid=False
)

plt.suptitle("")

plt.title(
    "Monthly Charges: Churned vs Retained Customers",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Customer Churn Status")
plt.ylabel("Monthly Charges")
plt.xticks(
    [1, 2],
    ["Retained (No Churn)", "Churned"]
)

plt.tight_layout()
plt.show()


# Graph 5: Churn Rate Across Tenure Groups

tenure_bins = [-1, 12, 24, 48, 72]

tenure_labels = [
    "0–12 months",
    "13–24 months",
    "25–48 months",
    "49–72 months"
]

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=tenure_bins,
    labels=tenure_labels
)

tenure_group_churn = (
    df.groupby("TenureGroup", observed=False)["Churn"]
    .apply(lambda values: (values == "Yes").mean() * 100)
)

plt.figure(figsize=(9, 5))

ax = tenure_group_churn.plot(kind="bar")

plt.title(
    "Churn Rate Across Tenure Groups",
    fontsize=16,
    fontweight="bold",
    pad=15
)
plt.xlabel("Customer Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.25)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3)

plt.tight_layout()
plt.show()

df = df.drop(columns=["TenureGroup"])


# 4. Prepare Features and Target

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]


# 5. Split the Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 6. Encode Categorical Features

X_train = pd.get_dummies(
    X_train,
    drop_first=True,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    drop_first=True,
    dtype=int
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# 7. Handle Missing Values

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# 8. Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# 9. Define Five Machine Learning Models

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=11
    ),

    "SVM": SVC(
        class_weight="balanced",
        random_state=42
    )
}


# 10. Train and Evaluate Models

results = []

predictions_by_model = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    predictions_by_model[name] = predictions

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Balanced Accuracy": balanced_accuracy
    })


# 11. Compare Model Performance

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
)

print("\n" + "=" * 90)
print("MACHINE LEARNING MODEL COMPARISON")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-score": "{:.4f}".format,
            "Balanced Accuracy": "{:.4f}".format
        }
    )
)


# Graph 6: Performance Comparison of Five ML Models

results_df.set_index("Model")[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Balanced Accuracy"
    ]
].plot(
    kind="bar",
    figsize=(13, 6)
)

plt.title(
    "Performance Comparison of Five ML Models",
    fontsize=16,
    fontweight="bold",
    pad=15
)
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# 12. Identify the Best Model Based on F1-score

best_model_name = results_df.iloc[0]["Model"]

best_f1 = results_df.iloc[0]["F1-score"]

best_predictions = predictions_by_model[best_model_name]

print("\n" + "=" * 60)
print("FINAL MODEL SELECTION")
print("=" * 60)

print("Best Model:", best_model_name)

print(f"Best F1-score: {best_f1:.4f}")


# Graph 7: Confusion Matrix — Best Performing Model

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    labels=[0, 1],
    display_labels=["No Churn", "Churn"],
    values_format="d"
)

plt.title(
    f"Confusion Matrix — Best Performing Model\n{best_model_name}",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.tight_layout()
plt.show()


# 13. Classification Reports for All Models

for name, predictions in predictions_by_model.items():

    print("\n" + "=" * 60)
    print(f"CLASSIFICATION REPORT: {name}")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )


# Graph 8: Precision vs Recall — Five-Model Comparison

plt.figure(figsize=(9, 6))

plt.scatter(
    results_df["Recall"],
    results_df["Precision"],
    s=120
)

for _, row in results_df.iterrows():

    plt.annotate(
        row["Model"],
        (row["Recall"], row["Precision"]),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=9
    )

plt.title(
    "Precision vs Recall: Five-Model Comparison",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel("Recall (Churners Identified)")
plt.ylabel("Precision (Correct Churn Predictions)")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# 14. Display the Best Model's Performance

best_result = results_df.iloc[0]

print("\nBest Model Performance:")

print(f"Accuracy:          {best_result['Accuracy']:.4f}")
print(f"Precision:         {best_result['Precision']:.4f}")
print(f"Recall:            {best_result['Recall']:.4f}")
print(f"F1-score:          {best_result['F1-score']:.4f}")
print(f"Balanced Accuracy: {best_result['Balanced Accuracy']:.4f}")


# 15. Final Conclusion

print("\nPROJECT CONCLUSION")

print(
    f"Among the five evaluated algorithms, {best_model_name} "
    f"achieved the highest F1-score on the test dataset."
)

print(
    "The models were compared using accuracy, precision, recall, "
    "F1-score, and balanced accuracy."
)

print(
    "F1-score balances precision and recall, while balanced "
    "accuracy considers recall for both churn classes."
)