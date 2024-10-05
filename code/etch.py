import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler, TargetEncoder, LabelEncoder
from sklearn.model_selection import cross_val_predict
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import Bunch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold

import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


occupation_mapping = {
    "00": "?",
    "01": "Administrators and Officials, Public Administration",
    "02": "Other Executive, Administrators, and Managers",
    "03": "Management Related Occupations",
    "04": "Engineers",
    "05": "Mathematical and Computer Scientists",
    "06": "Natural Scientists",
    "07": "Health Diagnosing Occupations",
    "08": "Health Assessment and Treating Occupations",
    "09": "Teachers, College and University",
    "10": "Teachers, Except College and University",
    "11": "Lawyers and Judges",
    "12": "Other Professional Specialty Occupations",
    "13": "Health Technologists and Technicians",
    "14": "Engineering and Science Technicians",
    "15": "Technicians, Except Health, Engineering, and Science",
    "16": "Supervisors and Proprietors, Sales Occupations",
    "17": "Sales Representatives, Finance, and Business Service",
    "18": "Sales Representatives, Commodities, Except Retail",
    "19": "Sales Workers, Retail and Personal Services",
    "20": "Sales Related Occupations",
    "21": "Supervisors - Administrative Support",
    "22": "Computer Equipment Operators",
    "23": "Secretaries, Stenographers, and Typists",
    "24": "Financial Records, Processing Occupations",
    "25": "Mail and Message Distributing",
    "26": "Other Administrative Support Occupations, Including Clerical",
    "27": "Private Household Service Occupations",
    "28": "Protective Service Occupations",
    "29": "Food Service Occupations",
    "30": "Health Service Occupations",
    "31": "Cleaning and Building Service Occupations",
    "32": "Personal Service Occupations",
    "33": "Mechanics and Repairers",
    "34": "Construction Trades",
    "35": "Other Precision Production Occupations",
    "36": "Machine Operators and Tenders, Except Precision",
    "37": "Fabricators, Assemblers, Inspectors, and Samplers",
    "38": "Motor Vehicle Operators",
    "39": "Other Transportation Occupations and Material Moving",
    "40": "Construction Laborer",
    "41": "Freight, Stock and Material Handlers",
    "42": "Other Handlers, Equipment Cleaners, and Laborers",
    "43": "Farm Operators and Managers",
    "44": "Farm Workers and Related Occupations",
    "45": "Forestry and Fishing Occupations",
    "46": "Armed Forces last job, currently unemployed"
}


# This was very wrong.
detailed_industry_mapping = {
    "00": "?",
    "01": "Agriculture Service",
    "02": "Other Agriculture",
    "03": "Mining",
    "04": "Construction",
# Manufacturing (Durable Goods)
    "05": "Lumber and wood products, except furniture",
    "06": "Furniture and fixtures",
    "07": "Stone clay, glass, and concrete product",
    "08": "Primary metals",
    "09": "Fabricated metal",
    "10": "Not specified metal industries",
    "11": "Machinery, except electrical",
    "12": "Electrical machinery, equipment, and supplies",
    "13": "Motor vehicles and equipment",
    "14": "Aircraft and parts",
    "15": "Other transportation equipment",
    "16": "Professional and photographic equipment, and watches",
    "17": "Toys, amusements, and sporting goods",
    "18": "Miscellaneous and not specified manufacturing industries",
# Manufacturing (Nondurable Goods)
    "19": "Food and kindred products",
    "20": "Tobacco manufactures",
    "21": "Textile mill products",
    "22": "Apparel and other finished textile products",
    "23": "Paper and allied products",
    "24": "Printing, publishing and allied industries",
    "25": "Chemicals and allied products",
    "26": "Petroleum and coal products",
    "27": "Rubber and miscellaneous plastics products",
    "28": "Leather and leather products",
    "29": "Transportation",
    "30": "Communications",
    "31": "Utilities and Sanitary Services",
    "32": "Wholesale Trade",
    "33": "Eating and drinking places",
    "34": "Other Retail Trade",
    "35": "Banking and Other Finance",
    "36": "Insurance and Real Estate",
    "37": "Private Household Services",
    "38": "Business Services",
    "39": "Repair Services",
    "40": "Personal Services, Except Private Household",
    "41": "Entertainment and Recreation Services",
    "42": "Hospitals",
    "43": "Health Services, Except Hospitals",
    "44": "Educational Services",
    "45": "Social Services",
    "46": "Other Professional Services",
    "47": "Forestry and Fisheries",
    "48": "Justice, Public Order and Safety",
    "49": "Administration of Human Resource Programs",
    "50": "National Security and Internal Affairs",
    "51": "Other Public Administration",
    "52": "Armed Forces last job, currently unemployed"
}


def load_census(codes=True):
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

    if not codes:
        # The formatting was strange so the descriptions may be wrong.
        train_df['detailed industry recode'] = train_df['detailed industry recode'].astype(str).str.zfill(2).map(detailed_industry_mapping)
        test_df['detailed industry recode'] = test_df['detailed industry recode'].astype(str).str.zfill(2).map(detailed_industry_mapping)
        train_df['detailed occupation recode'] = train_df['detailed occupation recode'].astype(str).str.zfill(2).map(occupation_mapping)
        test_df['detailed occupation recode'] = test_df['detailed occupation recode'].astype(str).str.zfill(2).map(occupation_mapping)


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


def create_education_pipeline():
    ordered_categories = [
        ' Children',
        ' Less than 1st grade', 
        ' 1st 2nd 3rd or 4th grade', 
        ' 5th or 6th grade', 
        ' 7th and 8th grade', 
        ' 9th grade', 
        ' 10th grade', 
        ' 11th grade', 
        ' 12th grade no diploma', 
        ' High school graduate', 
        ' Some college but no degree', 
        ' Associates degree-occup /vocational', 
        ' Associates degree-academic program', 
        ' Bachelors degree(BA AB BS)', 
        ' Masters degree(MA MS MEng MEd MSW MBA)',
        ' Doctorate degree(PhD EdD)',
        ' Prof school degree (MD DDS DVM LLB JD)',
    ]

    encoder = OrdinalEncoder(categories=[ordered_categories])

    # Education is partially orderable.
    return Pipeline(steps=[
        ('ordinal', encoder),
        ('scaler', MinMaxScaler())
    ])


def create_census_preprocessor(scalers, ones):
    return ColumnTransformer(
        transformers = [
            ('scaler', StandardScaler(), scalers),
            ('one', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ones),
            ('education', create_education_pipeline(), ['education'])
        ]
    )


# This file is all about the hard coding.
census_quantitative = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "weeks worked in year"
]

census_qualitative = [
    'class of worker',
    'detailed industry recode',
    'detailed occupation recode',
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
    'tax filer stat',
    'region of previous residence',
    'state of previous residence',
    'detailed household and family stat',
    'detailed household summary in household',
    'migration code-change in msa',
    'migration code-change in reg',
    'migration code-move within reg',
    'live in this house 1 year ago',
    'migration prev res in sunbelt',
    'family members under 18',
    'country of birth father',
    'country of birth mother',
    'country of birth self',
    'citizenship',
    'own business or self employed',
    "fill inc questionnaire for veteran's admin",
    'veterans benefits',
    'year'
]

def create_census_pipeline(estimator, quantitative=census_quantitative, qualitative=census_qualitative, transformer=None):
    return Pipeline(
        steps=[
                ('preprocessor', transformer if transformer else create_census_preprocessor(quantitative, qualitative)),
                ('estimator', estimator)
        ]
    )


def create_resample_pipeline(estimator, sampler):
    return ImbPipeline(steps=[
        ('resample', sampler),
        ('estimator', estimator)
    ])


def teska(estimator, census, sampler=None, transformer=None):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(census.y_train)
    y_test_encoded = encoder.transform(census.y_test)

    classifier = estimator
    census_pipeline = create_census_pipeline(classifier, transformer=transformer)
    pipeline = census_pipeline
    if sampler:
        resample_pipeline = create_resample_pipeline(census_pipeline, sampler)
        pipeline = resample_pipeline

    pipeline.fit(census.X_train, y_train_encoded)

    feature_names = census_pipeline.named_steps["preprocessor"].get_feature_names_out()
    X_train_encoded = pd.DataFrame(census_pipeline.named_steps["preprocessor"].transform(census.X_train), columns=feature_names)
    X_test_encoded = pd.DataFrame(census_pipeline.named_steps["preprocessor"].transform(census.X_test), columns=feature_names)

    return pipeline, (X_train_encoded, y_train_encoded), (X_test_encoded, y_test_encoded)


def predict_probabilities(estimator, X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')


def ys(y, y_probabilities, threshold=0.5):
    y_predictions = (y_probabilities[:, 1] >= threshold).astype(int)
    print(classification_report(y, y_predictions))
    print(confusion_matrix(y, y_predictions))


def plot_learning_curve(estimator, X, y, n_splits=5, random_state=42, scoring='f1', save_path=None):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring=scoring
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o')
    plt.plot(train_sizes, val_mean, label='Validation Score', color='green', marker='o')

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='green', alpha=0.2)

    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    if isinstance(scoring, str):
        plt.ylabel(scoring)
    plt.legend(loc='best')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def scale_target_transform(quantitative, qualitative, random_state=42):
    return ColumnTransformer(
        transformers = [
            ('quantitative', StandardScaler(), quantitative),
            ('qualitative', TargetEncoder(random_state=42), qualitative)
        ]
    )


# No time to work on this :)
class SelectFromCollection():
    def __init__(self, ax, collection, alpha_other=0.3, onscreen=None):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

        self.onscreen = onscreen

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]

        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        if self.onscreen:
            self.onscreen(self.ind) 
        self.canvas.draw_idle()
