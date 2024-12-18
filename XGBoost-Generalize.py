#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# Necessary Imports
from azureml.core import Workspace, Dataset, Run
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve 
from imblearn.under_sampling import RandomUnderSampler
from boruta import BorutaPy
from deap import base, creator, tools, algorithms
import random
import xgboost as xgb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
import boruta
import gc  # For garbage collection

# Suppress warnings
warnings.filterwarnings('ignore')

# Fix the np.int issue in BorutaPy
boruta.BorutaPy._fit.__globals__['np.int'] = int

# Connect to the Azure workspace
ws = Workspace.from_config()

# Access the Tabular Dataset
dataset = Dataset.get_by_name(ws, name="Epitope", version="1")

# Get the run context (Important to log data to Azure ML)
run = Run.get_context()

# Load the dataset into a Pandas DataFrame
Epitope_df = dataset.to_pandas_dataframe()

# Initial class distribution and log it
initial_class_distribution = Epitope_df['Class'].value_counts()
print("Initial class distribution:\n", initial_class_distribution)
run.log("Initial class distribution", initial_class_distribution.to_dict())

# Visualize and log the class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=Epitope_df)
plt.title('Initial Class Distribution')
run.log_image('Initial Class Distribution', plot=plt)
plt.close()

# Select features and target
X = Epitope_df.drop(columns=['Class'])
y = Epitope_df['Class'].replace(-1, 0)

# Identify numerical features
numerical_cols = X.select_dtypes(include=[np.float64, np.int64])

# Compute and log the correlation matrix
corr_matrix = numerical_cols.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.title("Correlation Matrix of Numeric Features")
run.log_image("Correlation Matrix of Numeric Features", plot=plt)
plt.close()

# Scale numerical features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(numerical_cols)
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Handle class imbalance using majority undersampling
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

# Log and visualize class distribution after undersampling
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train_res)
plt.title('Class Distribution After Undersampling')
run.log_image('Class Distribution After Undersampling', plot=plt)
plt.close()

print("Class distribution after undersampling:\n", pd.Series(y_train_res).value_counts())

# ------------------------------------------------
# Sampling the dataset for Boruta and GA
# ------------------------------------------------
# Sample 20% of the resampled dataset (adjust sample size based on memory constraints)
X_sample, _, y_sample, _ = train_test_split(X_train_res, y_train_res, test_size=0.8, stratify=y_train_res, random_state=42)

# Apply Boruta Feature Selection on the sampled data
xgb_clf = xgb.XGBClassifier(n_jobs=-1, random_state=42)
boruta_selector = BorutaPy(xgb_clf, n_estimators='auto', max_iter=50, random_state=42)
boruta_selector.fit(X_sample.values, y_sample.values)

# Select features confirmed by Boruta on the sample
selected_features = X_train_res.columns[boruta_selector.support_].to_list()

# Apply the selected features to the full dataset (X_train_res and X_test)
X_train_boruta = X_train_res[selected_features]
X_test_boruta = X_test[selected_features]

# Log the feature selection process
num_features_before_boruta = X_train.shape[1]
num_features_after_boruta = len(selected_features)
boruta_reduction_percent = ((num_features_before_boruta - num_features_after_boruta) / num_features_before_boruta) * 100
run.log("Number of features before Boruta", num_features_before_boruta)
run.log("Number of features after Boruta", num_features_after_boruta)
run.log("Percentage reduction after Boruta", boruta_reduction_percent)

# Visualize and log feature importance after Boruta
xgb_clf.fit(X_train_boruta, y_train_res)
feature_importances_boruta = xgb_clf.feature_importances_
importance_df_boruta = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances_boruta}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_boruta)
plt.title('Feature Importance (After Boruta)')
run.log_image("Feature Importance After Boruta", plot=plt)
plt.close()

# Ensure that X_train_boruta and y_train_res have the same number of samples
assert len(X_train_boruta) == len(y_train_res), "Sample size mismatch after feature selection."

# Batch size for batch processing
batch_size = 5000

# Batch size for batch processing
batch_size = 5000

# Define a function to train XGBoost in batches using `xgb.train()` for continuation training
def batch_train_xgboost(X_train_boruta, y_train_res, model, num_round=10):
    # Convert the entire training dataset into DMatrix (XGBoost's data structure)
    dtrain = xgb.DMatrix(X_train_boruta, label=y_train_res)
    
    # Specify the training parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Set up a watchlist to monitor training performance
    watchlist = [(dtrain, 'train')]
    
    # Loop through the data in batches for training
    for start in range(0, len(X_train_boruta), batch_size):
        end = min(start + batch_size, len(X_train_boruta))
        X_batch = X_train_boruta.iloc[start:end]
        y_batch = y_train_res.iloc[start:end]
        
        # Convert the batch to DMatrix
        dtrain_batch = xgb.DMatrix(X_batch, label=y_batch)
        
        # Continue training with the current model state
        model = xgb.train(params, dtrain_batch, num_boost_round=num_round, evals=watchlist, xgb_model=model)
        
        # Invoke garbage collection after each batch
        del X_batch, y_batch, dtrain_batch
        gc.collect()

    return model

# Initialize the XGBoost model
xgb_model = None  # Start with a fresh model (None)

# Train XGBoost model with batch processing
xgb_model = batch_train_xgboost(X_train_boruta, y_train_res, xgb_model)

# Define the evaluation function for GA with XGBoost
def evaluate(individual):
    selected_features = [index for index, value in enumerate(individual) if value == 1]
    if len(selected_features) == 0:
        return 0.0,

    X_train_selected = X_train_boruta.iloc[:, selected_features]

    # Ensure the correct alignment before splitting
    assert len(X_train_selected) == len(y_train_res), "Mismatch between selected features and labels."

    skf = StratifiedKFold(n_splits=5)
    accuracy = []

    for train_index, test_index in skf.split(X_train_selected, y_train_res):
        X_train_cv, X_test_cv = X_train_selected.iloc[train_index], X_train_selected.iloc[test_index]
        y_train_cv, y_test_cv = y_train_res.iloc[train_index], y_train_res.iloc[test_index]

        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(n_jobs=-1, random_state=42)
        xgb_model.fit(X_train_cv, y_train_cv)

        accuracy.append(xgb_model.score(X_test_cv, y_test_cv))  # Accuracy

    return np.mean(accuracy),

# Genetic Algorithm (GA) setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(selected_features))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize population
population = toolbox.population(n=20)

# Parameters for early stopping
patience = 3  # Stop if no improvement after 2 generations
best_fitness = -np.inf  # Initialize with the worst possible fitness
generations_without_improvement = 0  # Count how many generations have passed without improvement

# Run GA
NGEN = 5  # Maximum number of generations
ga_accuracies = []  # Track accuracy across generations

for gen in range(NGEN):
    print(f"Generation {gen + 1}")
    
    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))
    
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Replace the old population with the new offspring
    population[:] = offspring

    # Find the best individual from the current generation
    best_individual = tools.selBest(population, k=1)[0]
    best_individual_fitness = toolbox.evaluate(best_individual)[0]
    ga_accuracies.append(best_individual_fitness)

    # Check for improvement
    if best_individual_fitness > best_fitness:
        print(f"New best fitness found: {best_individual_fitness}")
        best_fitness = best_individual_fitness
        generations_without_improvement = 0  # Reset counter
    else:
        generations_without_improvement += 1
        print(f"No improvement. {generations_without_improvement} generation(s) without improvement.")

    # Early stopping condition
    if generations_without_improvement >= patience:
        print(f"Stopping early at generation {gen + 1} due to no improvement for {patience} generations.")
        break

# Visualize and log accuracy over generations
plt.figure(figsize=(10, 6))
plt.plot(range(len(ga_accuracies)), ga_accuracies, marker='o', linestyle='--')
plt.title('GA Accuracy Over Generations with Early Stopping')
plt.xlabel('Generations')
plt.ylabel('Accuracy')
run.log_image("GA Accuracy Over Generations with Early Stopping", plot=plt)
plt.close()

# Extract the best individual
best_individual = tools.selBest(population, k=1)[0]
selected_features_ga = [index for index, value in enumerate(best_individual) if value == 1]

# Calculate percentage reduction after GA feature selection
num_features_after_ga = len(selected_features_ga)
ga_reduction_percent = ((len(selected_features) - num_features_after_ga) / len(selected_features)) * 100
total_reduction_percent = ((X_train.shape[1] - num_features_after_ga) / X_train.shape[1]) * 100

# Log reduction details
run.log("Number of features before GA", len(selected_features))
run.log("Number of features after GA", num_features_after_ga)
run.log("Percentage reduction after GA", ga_reduction_percent)
run.log("Total percentage reduction", total_reduction_percent)

# Train the final XGBoost model using the selected features
X_train_final = X_train_boruta.iloc[:, selected_features_ga]
X_test_final = X_test_boruta.iloc[:, selected_features_ga]

# Define the parameter space for Bayesian optimization
param_space = {
    'n_estimators': (50, 300),  # Number of trees
    'max_depth': (3, 10),  # Maximum depth of the tree
    'learning_rate': (0.01, 0.3, 'uniform'),  # Step size shrinkage
    'subsample': (0.5, 1.0, 'uniform'),  # Fraction of samples to be used for fitting
    'colsample_bytree': (0.5, 1.0, 'uniform')  # Fraction of features to be used for each tree
}

# Initialize Bayesian optimization with XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)

opt = BayesSearchCV(
    xgb_model,
    param_space,
    n_iter=10,  # Number of parameter settings sampled
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)

# Fit the Bayesian optimization model
opt.fit(X_train_final, y_train_res)

# Get the best hyperparameters
best_hyperparameters = opt.best_params_
print("Best Hyperparameters (Bayesian):", best_hyperparameters)

# Extract the best model and retrain on the full dataset
best_xgb_model = opt.best_estimator_
best_xgb_model.fit(X_train_final, y_train_res)

# Predict probabilities for the test set
y_pred_prob = best_xgb_model.predict_proba(X_test_final)[:, 1]

# Tune the threshold for optimal precision-recall balance
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best Threshold: {best_threshold}")

# Predict the final labels using the best threshold
y_pred = (y_pred_prob >= best_threshold).astype("int32")

# Evaluate the final model
classification_report_str = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_str)
run.log("Classification Report", classification_report_str)

# Calculate AUC-ROC Score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC Score: {auc_score:.3f}")
run.log("AUC-ROC Score", auc_score)

# Plot and Log the AUC-ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC-ROC Curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
run.log_image("AUC-ROC Curve", plot=plt)
plt.close()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
run.log("Confusion Matrix", conf_matrix)

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
run.log_image("Confusion Matrix", plot=plt)
plt.close()

# Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='purple', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
run.log_image("Precision-Recall Curve", plot=plt)
plt.close()

# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.3f}")
run.log("Matthews Correlation Coefficient", mcc)

# Distribution of Predicted Probabilities (Histogram)
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_prob, kde=True)
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability')
run.log_image("Distribution of Predicted Probabilities", plot=plt)
plt.close()

# Feature importance after GA
feature_importances_ga = best_xgb_model.feature_importances_
importance_df_ga = pd.DataFrame({
    'Feature': X_train_boruta.columns[selected_features_ga],
    'Importance': feature_importances_ga
}).sort_values(by='Importance', ascending=False)

# Visualize and log feature importance after GA
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_ga)
plt.title('Feature Importance of Selected Features (After GA)')
run.log_image("Feature Importance After GA", plot=plt)
plt.close()

# Additional Visualizations

# 1. Distribution of Features After Boruta (Histograms)
for feature in selected_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(X_train_boruta[feature], kde=True)
    plt.title(f'Distribution of {feature} (After Boruta)')
    run.log_image(f'Distribution of {feature} (After Boruta)', plot=plt)
    plt.close()

# 2. Boxplot of Scaled Features (After Scaling)
plt.figure(figsize=(15, 8))
sns.boxplot(data=X_scaled_df, orient="h")
plt.title("Boxplot of Scaled Features (After MinMax Scaling)")
run.log_image("Boxplot of Scaled Features", plot=plt)
plt.close()

# 3. Correlation Heatmap After Boruta
plt.figure(figsize=(12, 10))
sns.heatmap(X_train_boruta.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap (After Boruta)")
run.log_image("Feature Correlation Heatmap (After Boruta)", plot=plt)
plt.close()

# Complete the run
run.complete()


