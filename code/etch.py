import pandas as pd
from sklearn.utils import Bunch


def load_census():
    feature_names = [
        'age',
        'class of worker',
        'detailed industry recode',
        'detailed occupation recode',
        'education',
        'wage per hour',
        'enroll in edu inst last wk',
        'marital stat',
        'major industry code',
        'major occupation code',
        'race',
        'hispanic origin',
        'sex',
        'member of a labor union',
        'reason for unemployment',
        'full or part time employment stat',
        'capital gains',
        'capital losses',
        'dividends from stocks',
        'tax filer stat',
        'region of previous residence',
        'state of previous residence',
        'detailed household and family stat',
        'detailed household summary in household',
        'instance weight',
        'migration code-change in msa',
        'migration code-change in reg',
        'migration code-move within reg',
        'live in this house 1 year ago',
        'migration prev res in sunbelt',
        'num persons worked for employer',
        'family members under 18',
        'country of birth father',
        'country of birth mother',
        'country of birth self',
        'citizenship',
        'own business or self employed',
        "fill inc questionnaire for veteran's admin",
        'veterans benefits',
        'weeks worked in year',
        'year'
    ]

    continuous_features=[
        "age",
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "instance weight",
        "num persons worked for employer",
        "weeks worked in year"
    ]

    nominal_features=list(set(feature_names) - set(continuous_features))

    target = "income"

    names = feature_names.copy()
    names.append(target)

    train_df = pd.read_csv('../data/census/census-income.data', names=names, index_col=False)
    test_df = pd.read_csv('../data/census/census-income.test', names=names, index_col=False)

    return Bunch(
        y_train=train_df["income"],
        X_train=train_df.drop(columns=["income"], axis=1),
        y_test=test_df["income"],
        X_test=test_df.drop(columns=["income"], axis=1),
        feature_names=feature_names,
        target=target,
        nominal_features=nominal_features,
        continuous_features=continuous_features
    )
