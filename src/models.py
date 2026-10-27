"""Model pipelines and hyper-parameter grids.

Every model is a single Pipeline (preprocessing -> [SMOTE] -> classifier), so
scaling, one-hot encoding and SMOTE are always fitted on training rows only,
including inside every cross-validation fold. The same fitted pipeline is
what gets saved, so training and prediction use identical preprocessing.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .data_loader import CATEGORICAL_COLS, NUMERIC_COLS

RANDOM_STATE = 42


def make_preprocess(numeric=NUMERIC_COLS, categorical=CATEGORICAL_COLS):
    parts = [("num", StandardScaler(), list(numeric))]
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)))
    return ColumnTransformer(parts)


def binary_models(y_train):
    """Return {name: (pipeline, param_grid)} for the binary task.

    RF / XGBoost / SVM keep the configuration of the original notebook (so the
    original 70/30 result can be reproduced). Two logistic-regression baselines
    are added: magnitude-only tests how much the extra features really add.
    The SVM is used without probability=True (deprecated in scikit-learn 1.9);
    its decision_function score is used for ROC-AUC / PR-AUC instead, which
    gives the same ranking-based metrics.
    """
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)  # computed from training labels only
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    return {
        "Dummy (majority)": (
            Pipeline([("prep", make_preprocess()),
                      ("clf", DummyClassifier(strategy="most_frequent"))]),
            None,
        ),
        "LogReg (magnitude only)": (
            Pipeline([("prep", make_preprocess(numeric=["magnitude"], categorical=[])),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]),
            None,
        ),
        "LogReg (all features)": (
            Pipeline([("prep", make_preprocess()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]),
            {"clf__C": [0.1, 1, 10]},
        ),
        "Random Forest + SMOTE": (
            ImbPipeline([("prep", make_preprocess()), ("smote", smote),
                         ("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10],
             "clf__min_samples_leaf": [1, 2, 4]},
        ),
        "XGBoost (weighted)": (
            Pipeline([("prep", make_preprocess()),
                      ("clf", XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                            eval_metric="logloss", scale_pos_weight=spw))]),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, 7],
             "clf__learning_rate": [0.05, 0.1]},
        ),
        "SVM (RBF) + SMOTE": (
            ImbPipeline([("prep", make_preprocess()), ("smote", smote),
                         ("clf", SVC(kernel="rbf", class_weight="balanced"))]),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.1, 1]},
        ),
    }


def multiclass_model(y_train):
    """XGBoost + SMOTE for the 3-level damage target (fixed settings, as in the original).

    SMOTE needs k_neighbors < size of the smallest class in the training fold.
    """
    counts = [int((y_train == c).sum()) for c in (0, 1, 2)]
    k = max(1, min(5, min(counts) - 1))
    return ImbPipeline([
        ("prep", make_preprocess()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k)),
        ("clf", XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss")),
    ])
