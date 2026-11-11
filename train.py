"""Train and persist Titanic survival classification models."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

DATASET_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
MODELS_DIR = Path(__file__).parent / "models"
FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def load_and_prepare_data(source: str = DATASET_URL) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Titanic dataset and apply the notebook preprocessing steps."""
    titanic = pd.read_csv(source)
    titanic = titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)

    titanic["Sex"] = titanic["Sex"].map({"male": 1, "female": 0})
    titanic["Embarked"] = titanic["Embarked"].map({"C": 1, "Q": 2, "S": 3})
    titanic = titanic.fillna(0)

    titanic["Sex"] = titanic["Sex"].astype(int)
    titanic["Embarked"] = titanic["Embarked"].astype(int)

    features = titanic[FEATURE_COLUMNS]
    labels = titanic["Survived"].astype(int)
    return features, labels


def train_and_save_models() -> dict[str, dict[str, float]]:
    """Train Logistic Regression and Decision Tree models and save them via joblib."""
    features, labels = load_and_prepare_data()
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.4, random_state=0
    )

    logistic_regressor = LogisticRegression(
        l1_ratio=0.5,
        max_iter=1000,
        solver="saga",
        random_state=0,
    )
    logreg_pipeline = Pipeline(
        steps=[
            ("add_poly_features", PolynomialFeatures()),
            ("logistic_regression", logistic_regressor),
        ]
    )
    logreg_pipeline.fit(features_train, labels_train)

    dtree_clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        random_state=0,
        min_samples_leaf=3,
        max_leaf_nodes=5,
    )
    dtree_clf.fit(features_train, labels_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg_pipeline, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(dtree_clf, MODELS_DIR / "decision_tree.joblib")

    metrics_summary = {}
    for name, model in [
        ("Binary Logistic Regression", logreg_pipeline),
        ("Decision Tree", dtree_clf),
    ]:
        preds = model.predict(features_test)
        metrics_summary[name] = {
            "accuracy": round(float(accuracy_score(labels_test, preds)), 4),
            "f1_score": round(float(f1_score(labels_test, preds)), 4),
        }

    return metrics_summary


if __name__ == "__main__":
    results = train_and_save_models()
    for model_name, scores in results.items():
        print(f"{model_name}: accuracy={scores['accuracy']}, f1={scores['f1_score']}")
