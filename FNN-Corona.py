#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Necessary Imports
from azureml.core import Workspace, Dataset, Run
import warnings
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, precision_recall_curve,roc_auc_score, roc_curve 
from imblearn.over_sampling import SMOTE  # Use SMOTE for oversampling
from boruta import BorutaPy
from deap import base, creator, tools, algorithms
import random
import xgboost as xgb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from skopt import BayesSearchCV
import tensorflow.keras.backend as K
import boruta
import tensorflow as tf

# Suppress warnings
warnings.filterwarnings('ignore')

# Fix the np.int issue in BorutaPy
boruta.BorutaPy._fit.__globals__['np.int'] = int

# Connect to the Azure workspace
ws = Workspace.from_config()

# Access the Tabular Dataset
dataset = Dataset.get_by_name(ws, name="Corona", version="1")

# Get the run context (Important to log data to Azure ML)
run = Run.get_context()

# Load the dataset into a Pandas DataFrame
Corona_df = dataset.to_pandas_dataframe()

# Initial class distribution and log it
initial_class_distribution = Corona_df['Class'].value_counts()
print("Initial class distribution:\n", initial_class_distribution)
run.log("Initial class distribution", initial_class_distribution.to_dict())

# Visualize and log the class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=Corona_df)
plt.title('Initial Class Distribution')
run.log_image('Initial Class Distribution', plot=plt)
plt.close()

# Select features and target
X = Corona_df.drop(columns=['Class'])
y = Corona_df['Class'].replace(-1, 0)

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

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Log and visualize class distribution after SMOTE
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train_res)
plt.title('Class Distribution After SMOTE')
run.log_image('Class Distribution After SMOTE', plot=plt)
plt.close()

print("Class distribution after SMOTE:\n", pd.Series(y_train_res).value_counts())

# Apply Boruta Feature Selection on the resampled data
xgb_clf = xgb.XGBClassifier(n_jobs=-1, random_state=42)
boruta_selector = BorutaPy(xgb_clf, n_estimators='auto', random_state=42)
boruta_selector.fit(X_train_res.values, y_train_res.values)

# Select features confirmed by Boruta
selected_features = X_train_res.columns[boruta_selector.support_].to_list()
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

# Define the evaluation function for GA with Feedforward Neural Network (FNN)
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

        fnn_model = create_fnn(input_dim=X_train_cv.shape[1])
        fnn_model.fit(X_train_cv, y_train_cv, epochs=5, verbose=0)  # Train for 5 epochs to save time

        accuracy.append(fnn_model.evaluate(X_test_cv, y_test_cv, verbose=0)[1])  # Accuracy

    return np.mean(accuracy),

# Define FNN model with 2 hidden layers, class weighting, and focal loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Cast y_true to the same dtype as y_pred (float32)
        y_true = tf.cast(y_true, dtype=tf.float32)

        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = -alpha_t * K.pow((1. - p_t), gamma) * K.log(p_t)
        return K.mean(loss)
    return focal_loss_fixed

def create_fnn(input_dim, units1=256, units2=128, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        Dense(units1, activation='relu', input_dim=input_dim),
        Dropout(dropout_rate),
        Dense(units2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Output layer
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
    return model

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

# Train the final Feedforward Neural Network (FNN) model using the selected features
X_train_final = X_train_boruta.iloc[:, selected_features_ga]
X_test_final = X_test_boruta.iloc[:, selected_features_ga]

# Bayesian optimization for hyperparameter tuning of FNN model (limited to 10 iterations)
fnn = KerasClassifier(build_fn=create_fnn, input_dim=X_train_final.shape[1], verbose=0)

param_space = {
    'units1': (64, 128),
    'units2': (32, 64),
    'dropout_rate': (0.1, 0.4, 'uniform'),
    'learning_rate': (0.0001, 0.01, 'log-uniform'),
    'batch_size': (8, 32),
    'epochs': (5, 20)
}

# Bayesian Optimization with only 10 iterations
opt = BayesSearchCV(
    fnn,
    param_space,
    n_iter=10,  # Reduced iterations to 10
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)

# Fit the Bayesian optimization model
opt.fit(X_train_final, y_train_res, class_weight={0: 1.0, 1: 5.0})

# Get the best hyperparameters
best_hyperparameters = opt.best_params_
print("Best Hyperparameters (Bayesian):", best_hyperparameters)

# Extract the best model and retrain on the full dataset
best_fnn_model = opt.best_estimator_
best_fnn_model.fit(X_train_final, y_train_res)

# Predict probabilities for the test set
y_pred_prob = best_fnn_model.predict_proba(X_test_final)[:, 1]

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
feature_importances_ga = best_fnn_model.model.layers[0].get_weights()[0].sum(axis=1)
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

