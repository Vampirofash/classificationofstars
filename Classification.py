import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight


# Настройка визуализаций
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"font.size": 12, "figure.figsize": (10, 6)})


def load_and_preprocess():
    """Загрузка и предобработка данных SIMBAD."""
    df = pd.read_csv(
        (
            r"D:\Books\_Документы и работы\Python\gaia_classifier"
            r"\simbad_region_dump.csv"
        ),
        dtype={
            "flux": str,
            "flux_err": str,
            "plx_value": str,
            "otype": str
        },
        low_memory=False,
    )

    numeric_cols = ["flux", "flux_err", "plx_value"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Очистка параллаксов регулярным выражением
    df["plx_value"] = (
        df["plx_value"]
        .astype(str)
        .str.extract(r"^(\d+\.?\d*)")
        .astype(float)
    )

    # Агрегация и преобразование данных о потоке
    flux_grouped = (
        df.groupby(["main_id", "flux.filter"])
        .agg({"flux": "mean", "flux_err": "mean"})
        .reset_index()
    )

    flux_pivot = flux_grouped.pivot(
        index="main_id",
        columns="flux.filter",
        values=["flux", "flux_err"],
    )
    flux_pivot.columns = [f"{col[0]}_{col[1]}" for col in flux_pivot.columns]
    flux_pivot = flux_pivot.reset_index()

    # Агрегация нечисловых данных
    non_flux = (
        df.groupby("main_id")
        .agg(
            otype=("otype", lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            plx_value=("plx_value", "mean"),
        )
        .reset_index()
    )

    merged = pd.merge(non_flux, flux_pivot, on="main_id", how="left")
    merged["otype"] = merged["otype"].fillna("Unknown").astype(str)

    merged["target"] = merged["otype"].apply(
        lambda x: (
            "White Dwarf"
            if x.startswith("WD")
            else "Standard Star" if x.strip() in ["*", "V*"] else "Peculiar/Other"
        )
    )

    return merged.dropna(subset=["target"])


def plot_distributions(data):
    """Визуализация распределений классов и признаков."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    sns.countplot(
        data=data,
        x="target",
        ax=ax[0],
        order=data["target"].value_counts().index,
    )
    ax[0].set_title("Распределение классов")
    ax[0].tick_params(axis="x", rotation=45)

    sns.histplot(
        data=data,
        x="plx_value",
        bins=30,
        ax=ax[1],
        kde=True,
    )
    ax[1].set_title("Распределение параллаксов")
    ax[1].set_xlabel("Параллакс (mas)")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    """Визуализация матрицы ошибок классификации."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=classes,
        cmap="Blues",
        ax=ax,
        colorbar=False,
        text_kw={"fontsize": 12},
    )
    plt.title("Матрица ошибок классификации", pad=20)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(pipeline, feature_names, top_n=15):
    """Визуализация топ-N значимых признаков с аннотациями."""
    importances = pipeline.named_steps["classifier"].feature_importances_
    
    valid_features = [fn for fn in feature_names if not pd.isnull(fn)]
    valid_importances = importances[:len(valid_features)]
    
    n_features = len(valid_features)
    top_n = min(top_n, n_features)
    indices = np.argsort(valid_importances)[-top_n:]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(top_n), valid_importances[indices], align="center")
    
    plt.yticks(range(top_n), [valid_features[i] for i in indices])
    plt.title(f"Топ-{top_n} значимых признаков")
    plt.xlabel("Относительная важность")

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.005,
            bar.get_y() + bar.get_height()/2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    
    plt.tight_layout()
    plt.show()


def main():
    """Основной пайплайн обучения и оценки модели."""
    data = load_and_preprocess()
    plot_distributions(data)

    features = ["plx_value"] + [
        col for col in data.columns if col.startswith("flux_")
    ]
    X = data[features]
    y = data["target"]

    missing = X.isnull().sum() / len(X) * 100
    X = X[missing[missing < 80].index]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = dict(zip(classes, class_weights))

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        ))
    ])

    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [None, 10, 20, 30, 50],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__bootstrap": [True, False],
        "classifier__ccp_alpha": [0.0, 0.01, 0.1]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("\nЛучшие параметры модели:")
    for param, value in search.best_params_.items():
        print(f"{param}: {value}")

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    joblib.dump(best_model, "random_forest_model.joblib")

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, best_model.named_steps["classifier"].classes_)
    plot_feature_importance(best_model, X.columns)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, pipeline.classes_)
    plot_feature_importance(pipeline, X.columns)


if __name__ == "__main__":
    main()