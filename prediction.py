import pandas as pd
import numpy as np
# from estimation import nan_var_replace_pre_feat, feature_engineering, nan_var_replace_post_feat, nan_var_median_post_feat_test

def nan_var_replace_pre_feat(df):

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['exp_financing'].fillna(df['inc_financing'] - df['prof_financing'], inplace=True)
    df['asst_current'].fillna(df['AR'] + df['cash_and_equiv'] + df['cf_operations'], inplace=True)
    df['asst_current'].fillna(df['asst_tot'] - (df['asst_fixed_fin'] + df['asst_tang_fixed'] + df['asst_intang_fixed']), inplace=True)
    df['cf_operations'].fillna(df['profit'], inplace=True)
    df['cf_operations'].fillna(df['cash_and_equiv'], inplace=True)
    df['eqty_tot'].fillna(df['asst_tot'] - (df['liab_lt'] + df['debt_bank_st'] + df['debt_bank_lt'] + df['debt_fin_st'] + df['debt_fin_lt'] + df['AP_st'] + df['AP_lt']), inplace=True)
    df['AP_st'].fillna(df['debt_st'], inplace=True)
    df['roe'].fillna(df['profit'] / df['eqty_tot'], inplace=True)
    df['roa'].fillna(df['profit'] / df['asst_tot'], inplace=True)
    df['days_rec'].fillna((df['AR'] / df['rev_operating']) * 365, inplace=True) 
    
    return df

def feature_engineering(df):

    df['debt_ratio'] = (df['liab_lt'] + df['debt_bank_st'] + df['debt_bank_lt'] + df['debt_fin_st'] + df['debt_fin_lt'] + df['AP_st'] + df['AP_lt']) / df['asst_tot']
    df['cash_return_assets'] = df['cf_operations'] / df['asst_tot']
    df['leverage'] = (df['debt_lt'] + df['debt_st']) / df['asst_tot']
    df['cash_ratio'] = (df['cash_and_equiv'] + df['cf_operations']) / (df['debt_st'] + df['AP_st'] + 1)        
    
    return df
    
def nan_var_replace_post_feat(df):
    
    df['debt_ratio'].fillna(df['asst_tot'] - df['eqty_tot'], inplace=True)

    return df

def nan_var_median_post_feat(df, preproc_params={}):

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for column in df.columns:
        if df[column].dtype == 'float64':
            df[column].fillna(preproc_params[column + '_median'], inplace=True)

    return df

def preprocessor(df, preproc_params={}):

    data = df.copy()

    # Imputes the variables before feature engineering (avoid inf or -inf)
    df_nan_impute = nan_var_replace_pre_feat(data)

    # Do feature engineering
    df_feat_eng = feature_engineering(df_nan_impute)

    # Impute post feature engineering (impute ratios)
    df_feat_eng_nan_impute = nan_var_replace_post_feat(df_feat_eng)

    # Required features
    features = [ 'roa',
                 'asst_tot',
                 'debt_ratio',
                 'cash_return_assets',
                 'leverage']
    
    # Drop unnecessary columns
    df_features = df_feat_eng_nan_impute.drop(columns = [col for col in df_feat_eng 
                                            if col not in features])
    
    # Final imputation using medians
    result_df = nan_var_median_post_feat(df_features[features], preproc_params)

    return result_df
    
def predictor_harness(new_df, model, preprocessor, output_csv, preproc_params={}):
    
    preprocessed_data = preprocessor(new_df, preproc_params)
    predictions = model.predict(preprocessed_data)

    def curve_func(x, a, b, c):
        return a * x**2 + b * x + c

    def calibrate_prob(raw_prob, params):
        calibrate_prob = [curve_func(x, *params) for x in raw_prob]
        return calibrate_prob
    
    calculated_coefficients = [5.86333743e+02, -7.31859846e+00, 1.75116159e-02]
        
    predictions = calibrate_prob(predictions, calculated_coefficients)
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_csv, index=False, header=False)

    return output_csv