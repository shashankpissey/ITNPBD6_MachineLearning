import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression # Import Logistic Regression
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint, uniform, loguniform # Import loguniform for C parameter
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_preprocessor(numerical_features, categorical_features):
    """Creates a preprocessing pipeline for numerical and categorical features."""
    # ... (preprocessor function remains the same as before) ...
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def run_multi_model_comparison_with_logistic_regression(file_path='wallacecommunications.csv'):
    # --- Data Loading and Splitting (Corrected with all features) ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    df = df.drop(columns=['ID'])
    df['job'].fillna('unknown', inplace=True)
    df['education'].fillna('unknown', inplace=True)
    df['days_since_last_contact_previous_campaign'].fillna(-1, inplace=True)
    df['arrears'].fillna('No_Info', inplace=True)

    categorical_features = [
        'town', 'country', 'job', 'married', 'education', 'housing', 
        'has_tv_package', 'last_contact', 'conn_tr', 
        'last_contact_this_campaign_month', 'outcome_previous_campaign',
        'arrears' 
    ]
    numerical_features = [
        'age', 'current_balance', 'this_campaign', 
        'days_since_last_contact_previous_campaign', 
        'contacted_during_previous_campaign',
        'last_contact_this_campaign_day' # All features included now
    ]
    target_variable = 'new_contract_this_campaign'

    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data splits: Training set size: {len(X_train)}, Test set size: {len(X_test)}\n")

    preprocessor = get_preprocessor(numerical_features, categorical_features)

    # --- Setup Models and Parameter Grids for Randomized Search ---
    
    # 1. Random Forest Configuration
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42))])
    rf_params = {
        'classifier__n_estimators': randint(100, 500), 'classifier__max_depth': [None] + list(randint(5, 30).rvs(size=5, random_state=42)),
        'classifier__min_samples_split': randint(2, 20), 'classifier__min_samples_leaf': randint(1, 10),
    }

    # 2. Support Vector Machine Configuration
    svm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(random_state=42, probability=True))])
    svm_params = {
        'classifier__C': uniform(loc=0.1, scale=10), 'classifier__gamma': uniform(loc=0.001, scale=1), 'classifier__kernel': ['rbf'] # Stick to rbf for simplicity
    }

    # 3. Neural Network (MLPClassifier) Configuration
    nn_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', MLPClassifier(random_state=42, max_iter=1500))])
    nn_params = {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'classifier__activation': ['tanh', 'relu'],
        'classifier__alpha': uniform(loc=0.0001, scale=0.01), 'classifier__learning_rate_init': [0.001, 0.01]
    }
    
    # 4. Logistic Regression Configuration
    # Note: Logistic Regression requires scaled numerical features, which our preprocessor provides.
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(random_state=42, solver='saga', max_iter=2000))]) # Use 'saga' solver for L1/L2 support
    lr_params = {
        'classifier__C': loguniform(0.01, 100), # Regularization strength parameter
        'classifier__penalty': ['l1', 'l2', 'elasticnet', None], # Type of regularization
        'classifier__l1_ratio': uniform(0, 1) # Only used if penalty='elasticnet'
    }


    models_to_tune = {
        'SVM': (svm_pipeline, svm_params, 20),         
        #'Neural Network': (nn_pipeline, nn_params, 30),
        'Logistic Regression': (lr_pipeline, lr_params, 30) # Add Logistic Regression to the mix
    }

    best_models_results = {}
    scoring_metric = 'f1_weighted' 

    print("--- Starting Hyperparameter Tuning for all models using Randomized Search CV ---")

    for name, (pipeline, params, n_iter) in models_to_tune.items():
        start_time = time.time()
        print(f"\n* Starting tuning for {name} with {n_iter} iterations...")
        
        random_search = RandomizedSearchCV(
            estimator=pipeline, 
            param_distributions=params, 
            n_iter=n_iter, 
            cv=StratifiedKFold(n_splits=5), 
            scoring=scoring_metric, 
            verbose=0, 
            random_state=42, 
            n_jobs=-1,
            # Add error_score='raise' if you want to debug parameter combinations that fail,
            # otherwise setting to np.nan will skip them.
            error_score=np.nan 
        )
        
        random_search.fit(X_train, y_train)
        end_time = time.time()

        print(f"Finished {name} tuning in {end_time - start_time:.2f} seconds.")
        print(f"Best CV Score ({scoring_metric}): {random_search.best_score_:.4f}")
        # print(f"Best Parameters: {random_search.best_params_}") # Optional print

        best_estimator = random_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        test_f1_score = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
        
        best_models_results[name] = {
            'best_estimator': best_estimator,
            'test_f1': test_f1_score,
            'best_cv_f1': random_search.best_score_
        }
        print(f"Test Set F1 Score (weighted): {test_f1_score:.4f}")


    # --- 4. Final Comparison ---
    print("\n\n" + "="*60)
    print("--- Final Model Comparison on Test Set (Weighted F1 Score) ---")
    print("="*60)
    
    for name, results in best_models_results.items():
        print(f"{name:<20}: F1 Score = {results['test_f1']:.4f}")
        
    # Determine the overall winner
    winner_name = max(best_models_results, key=lambda k: best_models_results[k]['test_f1'])
    print(f"\nOverall Best Model is the {winner_name}!")
    

if __name__ == "__main__":
    # Ensure you have the 'wallacecommunications.csv' file in the same directory
    run_multi_model_comparison_with_logistic_regression(file_path='wallacecommunications.csv')
