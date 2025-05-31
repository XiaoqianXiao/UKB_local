import numpy as np
import pandas as pd
from scipy.stats import norm


# Define KDM-BA calculation function
def kdm_calc(data, biomarkers, fit=None):
    """
    Calculate Klemera-Doubal Method Biological Age (KDM-BA).

    Parameters:
    - data: pandas DataFrame with biomarker columns and 'age' column
    - biomarkers: list of biomarker column names
    - fit: dict with 'slope' and 'intercept' for each biomarker (optional)

    Returns:
    - DataFrame with KDM-BA and advancement (KDM-BA - chronological age)
    """
    if fit is None:
        # Default fit parameters (example values, should be trained on NHANES III)
        fit = {
            bm: {'slope': 0.1, 'intercept': np.mean(data[bm]), 'sd': np.std(data[bm])}
            for bm in biomarkers
        }

    # Calculate KDM-BA
    kdm_ba = []
    for _, row in data.iterrows():
        age = row['age']
        num = 0
        denom = 0
        for bm in biomarkers:
            x = row[bm]
            k = fit[bm]['slope']
            q = fit[bm]['intercept']
            s = fit[bm]['sd']
            # KDM formula: (biomarker - intercept) / slope contributes to age
            num += ((x - q) / s) * (k / s)
            denom += (k / s) ** 2
        # Final KDM-BA: weighted average of biomarker contributions
        kdm = age + (num / denom) if denom != 0 else age
        kdm_ba.append(kdm)

    result = data.copy()
    result['kdm_ba'] = kdm_ba
    result['kdm_acceleration'] = result['kdm_ba'] - result['age']
    return {'data': result, 'fit': fit}


# Define PhenoAge calculation function
def phenoage_calc(data, biomarkers, fit=None, orig=True):
    """
    Calculate Phenotypic Age (PhenoAge) based on Levine et al. 2018.

    Parameters:
    - data: pandas DataFrame with biomarker columns and 'age' column
    - biomarkers: list of biomarker column names
    - fit: dict with coefficients for survival model (optional)
    - orig: bool, if True, use original PhenoAge coefficients

    Returns:
    - DataFrame with PhenoAge and advancement
    """
    if fit is None:
        # Example coefficients from Levine 2018 (simplified, should be trained)
        fit = {
            'albumin': -0.0336,
            'creatinine': 0.0095,
            'glucose': 0.1953,
            'lncrp': 0.0954,
            'lymph': -0.0120,
            'mcv': 0.0268,
            'rdw': 0.3306,
            'alp': 0.0019,
            'wbc': 0.0554,
            'intercept': -19.9067,
            'gamma': 0.0077
        }

    # Calculate mortality score
    mortality_score = []
    for _, row in data.iterrows():
        score = fit['intercept']
        for bm in biomarkers:
            score += fit[bm] * row[bm]
        mortality_score.append(score)

    # Convert to PhenoAge using Gompertz CDF
    gamma = fit['gamma']
    phenoage = []
    for score in mortality_score:
        # CDF(120, x) = 1 - exp(-gamma * exp(score))
        cdf_120 = 1 - np.exp(-gamma * np.exp(score))
        # Solve for age: CDF(120, age) = CDF(120, x)
        age_est = -np.log(1 - cdf_120) / gamma if cdf_120 < 1 else 120
        phenoage.append(age_est)

    result = data.copy()
    result['phenoage'] = phenoage
    result['phenoage_acceleration'] = result['phenoage'] - result['age']
    return {'data': result, 'fit': fit}


# Example usage
if __name__ == "__main__":
    # Sample data (mimicking NHANES structure)
    data = pd.DataFrame({
        'sampleID': [1, 2, 3],
        'age': [30, 40, 50],
        'albumin': [45, 42, 40],  # g/L
        'creatinine': [80, 85, 90],  # umol/L
        'glucose': [5.0, 5.5, 6.0],  # mmol/L
        'lncrp': [0.5, 0.7, 0.9],  # log mg/L
        'lymph': [30, 28, 26],  # %
        'mcv': [90, 92, 94],  # fL
        'rdw': [12.5, 13.0, 13.5],  # %
        'alp': [70, 75, 80],  # U/L
        'wbc': [6.0, 6.5, 7.0]  # 10^9/L
    })

    # Define biomarkers (same as BioAge package example)
    biomarkers = ['albumin', 'creatinine', 'glucose', 'lncrp', 'lymph', 'mcv', 'rdw', 'alp', 'wbc']

    # Calculate KDM-BA
    kdm_result = kdm_calc(data, biomarkers)
    print("KDM-BA Results:")
    print(kdm_result['data'][['sampleID', 'age', 'kdm_ba', 'kdm_acceleration']])

    # Calculate PhenoAge
    phenoage_result = phenoage_calc(data, biomarkers)
    print("\nPhenoAge Results:")
    print(phenoage_result['data'][['sampleID', 'age', 'phenoage', 'phenoage_acceleration']])