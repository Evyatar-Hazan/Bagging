import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    טוען את קובץ ה-CSV ומסנן תוצאות עם סטטוס PASSED או FAILED בלבד.
    מוסיף עמודת מטרה בינארית ועמודת קידוד לתסריט.
    """
    df = pd.read_csv(filepath)
    df = df[df["status"].isin(["PASSED", "FAILED"])].copy()

    # יצירת עמודת מטרה בינארית: FAILED=1, PASSED=0
    df["target"] = df["status"].map({"FAILED": 1, "PASSED": 0})

    # קידוד שם התסריט למספרים
    script_encoder = LabelEncoder()
    df["script_encoded"] = script_encoder.fit_transform(df["script_name"])

    return df


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    מאמן מודל RandomForest על תכונות execution_time ו-script_encoded.
    מחזיר את המודל המאומן.
    """
    X = df[["execution_time", "script_encoded"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model


def add_fail_probability(df: pd.DataFrame, model: RandomForestClassifier) -> pd.DataFrame:
    """
    מוסיף לעמודת df את הסתברות הכישלון עבור כל שורה.
    """
    df["fail_probability"] = model.predict_proba(df[["execution_time", "script_encoded"]])[:, 1]
    return df


def plot_top_risky_scripts(df: pd.DataFrame) -> None:
    """
    מציג גרף עמודות אינטראקטיבי עם התסריטים הכי מסוכנים לפי הסיכוי לכישלון.
    """
    # סינון הריצה האחרונה לכל תסריט לפי timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest_runs = df.sort_values("timestamp").groupby("script_name").tail(1)

    # בחירת 5 התסריטים עם הסיכוי הגבוה ביותר לכישלון
    top_risky = latest_runs[["script_name", "execution_time", "fail_probability", "status"]]
    top_risky = top_risky.sort_values("fail_probability", ascending=False).head(5)

    fig = px.bar(
        top_risky,
        x="script_name",
        y="fail_probability",
        color="status",
        title="📉 תסריטים עם סיכוי גבוה להיכשל",
        labels={"fail_probability": "סיכוי לכישלון", "script_name": "שם תסריט"},
        text="execution_time"
    )

    fig.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()


def plot_fail_probability_trend(df: pd.DataFrame, top_scripts: list) -> None:
    """
    מציג גרף קווי שמראה את שינוי סיכוי הכישלון לאורך זמן עבור תסריטים מובילים.
    """
    recent = df[df["script_name"].isin(top_scripts)]

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=recent, x="timestamp", y="fail_probability", hue="script_name", marker="o")
    plt.title("⏱ מגמת סיכוי כישלון לאורך זמן")
    plt.xlabel("תאריך")
    plt.ylabel("סיכוי לכישלון")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # --- טעינת הנתונים והכנתם ---
    data_path = "csv_reports/all_runs.csv"
    df = load_and_prepare_data(data_path)

    # --- אימון המודל ---
    model = train_model(df)

    # --- הוספת עמודת סיכוי כישלון ---
    df = add_fail_probability(df, model)

    # --- הצגת תסריטים בסיכון גבוה ---
    plot_top_risky_scripts(df)

    # --- הצגת מגמת סיכוי כישלון לאורך זמן עבור 5 התסריטים המסוכנים ביותר ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest_runs = df.sort_values("timestamp").groupby("script_name").tail(1)
    top_5_scripts = latest_runs.sort_values("fail_probability", ascending=False).head(5)["script_name"].tolist()

    plot_fail_probability_trend(df, top_5_scripts)


if __name__ == "__main__":
    main()
 