import pandas as pd
from sklearn.utils import Bunch


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


detailed_industry_mapping = {
    "00": "NIU(children)",
    "01": "Agriculture",
    "02": "Mining",
    "03": "Construction",
# Manufacturing
#   Durable Goods
    "04": "Lumber and Wood Products, except Furniture",
    "05": "Furniture and Fixtures",
    "06": "Stone, Clay, Glass, Concrete Products",
    "07": "Primary Metals",
    "08": "Fabricated Metals",
    "09": "Not Specified Metal Industries",
    "10": "Machinery, except Electrical",
    "11": "Electrical Machinery, Equipment, Supplies",
#   Transportation Equipment
      "12": "Motor Vehicles and Equipment",
#     Other Transportation Equipment
        "13": "Aircraft and Parts",
        "14": "Other Transportation Equipment",
    "15": "Professional and Photo Equipment, Watches",
    "16": "Toys, Amusements, and Sporting Goods",
    "17": "Miscellaneous and Not Specified",
#   Nondurable Goods
      "18": "Food and Kindred Products",
      "19": "Tobacco Manufactures",
      "20": "Textile Mill Products",
      "21": "Apparel and Other Finished Textile Products",
      "22": "Paper and Allied Products",
      "23": "Printing, Publishing, and Allied Industries",
      "24": "Chemicals and Allied Products",
      "25": "Petroleum and Coal Products",
      "26": "Rubber and Miscellaneous Plastics Products",
      "27": "Leather and Leather Products",
# Transportation, Communications, and Other Public Utilities
    "28": "Transportation",
#   Communication and Other Public Utilities
      "29": "Communication",
      "30": "Utilities and Sanitary Services",
#   Wholesale and Retail Trade
      "31": "Wholesale Trade",
      "32": "Retail Trade",
#   Finance, Insurance, and Real Estate
      "33": "Banking and Other Finance",
      "34": "Insurance and Real Estate",
#   Service
      "35": "Private Household",
#   Miscellaneous Services
#   Business and Repair Services
      "36": "Business Services",
      "37": "Repair Services",
    "38": "Personal Service except Private Household",
    "39": "Entertainment and Recreation Services",
#   Professional and Related Services
      "40": "Hospitals",
      "41": "Health Services, except Hospitals",
      "42": "Educational Services",
      "43": "Social Services",
      "44": "Other Professional Services",
    "45": "Forestry and Fisheries",
    "46": "Public Administration",
    "47": "Never Worked (WKSWORK=0)"
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

    if codes:
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
