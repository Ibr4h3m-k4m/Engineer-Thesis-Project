# Complete XGBoost Hyperparameter Optimization Script for Local Environment
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from scipy.optimize import differential_evolution
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
# Modified for local environment - all files in current directory
DATASET_PATH = '.'  # Current directory instead of Kaggle path
ensemble_dir = './xgboost_hpo_ensemble'  # Local directory
os.makedirs(ensemble_dir, exist_ok=True)
print(f"Created ensemble directory: {ensemble_dir}")

# ===== LOAD DATASETS =====
print("Loading datasets...")
try:
    X_train = pd.read_csv(f'{DATASET_PATH}/X_train.csv')
    X_val = pd.read_csv(f'{DATASET_PATH}/X_val.csv')
    X_test = pd.read_csv(f'{DATASET_PATH}/X_test.csv')
    y_train = pd.read_csv(f'{DATASET_PATH}/y_train.csv')
    y_val = pd.read_csv(f'{DATASET_PATH}/y_val.csv')
    y_test = pd.read_csv(f'{DATASET_PATH}/y_test.csv')
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset files. Make sure the following files are in the current directory:")
    print("- X_train.csv, X_val.csv, X_test.csv")
    print("- y_train.csv, y_val.csv, y_test.csv")
    print(f"Current directory: {os.getcwd()}")
    raise e

# Combine train and val for cross-validation
X_combined = pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True)
y_combined = pd.concat([y_train, y_val, y_test], axis=0, ignore_index=True)

# ===== FEATURE ENGINEERING =====
def engineer_advanced_features(X: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as the successful model"""
    X = X.copy()
    
    # Base features (8 BRO features)
    base_features = ['senderpseudo', 'posx', 'posy', 'posx_n', 'spdx', 'spdy', 'hedy', 'hedx_n']
    X_base = X[base_features]
    
    print("Applying feature engineering...")
    
    # 1. Inter-packet arrival time
    X_base['sender_sequence'] = X_base.groupby('senderpseudo').cumcount()
    X_base['inter_arrival_time'] = X_base.groupby('senderpseudo')['sender_sequence'].diff().fillna(1.0)
    
    # 2. Speed and acceleration features
    X_base['speed_magnitude'] = np.sqrt(X_base['spdx']**2 + X_base['spdy']**2)
    X_base['acceleration_x'] = X_base.groupby('senderpseudo')['spdx'].diff().fillna(0)
    X_base['acceleration_y'] = X_base.groupby('senderpseudo')['spdy'].diff().fillna(0)
    X_base['acceleration_magnitude'] = np.sqrt(X_base['acceleration_x']**2 + X_base['acceleration_y']**2)
    
    # 3. Position change features
    X_base['position_change_x'] = X_base.groupby('senderpseudo')['posx'].diff().fillna(0)
    X_base['position_change_y'] = X_base.groupby('senderpseudo')['posy'].diff().fillna(0)
    X_base['position_change_magnitude'] = np.sqrt(X_base['position_change_x']**2 + X_base['position_change_y']**2)
    
    # 4. Heading features
    X_base['heading_change'] = X_base.groupby('senderpseudo')['hedy'].diff().fillna(0)
    X_base['heading_magnitude'] = np.sqrt(X_base['hedx_n']**2 + X_base['hedy']**2)
    
    # 5. Behavioral consistency features
    X_base['speed_consistency'] = X_base.groupby('senderpseudo')['speed_magnitude'].transform('std').fillna(0)
    X_base['position_consistency'] = X_base.groupby('senderpseudo')['position_change_magnitude'].transform('std').fillna(0)
    
    # 6. Temporal features
    X_base['hour'] = (X_base['sender_sequence'] % 24)
    X_base['night_hours'] = X_base['hour'].between(22, 5, inclusive="left").astype(int)
    
    # 7. Interaction features
    X_base['speed_position_interaction'] = X_base['speed_magnitude'] * X_base['position_change_magnitude']
    X_base['inter_arrival_speed_ratio'] = X_base['inter_arrival_time'] / (X_base['speed_magnitude'] + 1e-6)
    
    X_base = X_base.fillna(0)
    return X_base

# Apply feature engineering
X_combined_engineered = engineer_advanced_features(X_combined)
print(f"Features after engineering: {X_combined_engineered.shape[1]}")

# ===== SENDER-AWARE DATA SPLITTING =====
def create_sender_aware_splits(X, y, train_val_ratio=0.85, random_state=42):
    """Split data by senders: 85% senders for train/val, 15% senders for test"""
    print("\n========== Sender-Aware Data Splitting ==========")
    
    senders = X['senderpseudo'].values
    unique_senders = np.unique(senders)
    print(f"Total unique senders: {len(unique_senders)}")
    
    np.random.seed(random_state)
    n_train_val_senders = int(train_val_ratio * len(unique_senders))
    
    shuffled_senders = np.random.permutation(unique_senders)
    train_val_senders = shuffled_senders[:n_train_val_senders]
    test_senders = shuffled_senders[n_train_val_senders:]
    
    print(f"Train/Val senders: {len(train_val_senders)} ({len(train_val_senders)/len(unique_senders)*100:.1f}%)")
    print(f"Test senders: {len(test_senders)} ({len(test_senders)/len(unique_senders)*100:.1f}%)")
    
    train_val_mask = np.isin(senders, train_val_senders)
    test_mask = np.isin(senders, test_senders)
    
    X_train_val = X[train_val_mask].reset_index(drop=True)
    y_train_val = y[train_val_mask].reset_index(drop=True)
    X_test_new = X[test_mask].reset_index(drop=True)
    y_test_new = y[test_mask].reset_index(drop=True)
    
    print(f"Train/Val data: {X_train_val.shape}")
    print(f"Test data (unseen senders): {X_test_new.shape}")
    
    return X_train_val, y_train_val, X_test_new, y_test_new, train_val_senders, test_senders

# Apply sender-aware splitting
X_train_val, y_train_val, X_test_unseen, y_test_unseen, train_val_senders, test_senders = create_sender_aware_splits(
    X_combined_engineered, y_combined.iloc[:, 0], train_val_ratio=0.85
)

# ===== PREPARE DATA FOR OPTIMIZATION =====
print("\n========== Preparing Data for Optimization ==========")

# Scale features
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)

# Calculate class weights
class_counts = np.bincount(y_train_val)
total_samples = len(y_train_val)
class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}

print("Class weights:", {i: f"{weight:.3f}" for i, weight in class_weights.items()})

# ===== GPU DETECTION =====
def check_xgboost_gpu():
    try:
        test_data = np.random.random((100, X_train_val_scaled.shape[1]))
        test_labels = np.random.randint(0, 6, 100)
        test_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'verbosity': 0,
            'n_estimators': 10,
            'objective': 'multi:softmax',
            'num_class': 6
        }
        xgb_test = XGBClassifier(**test_params)
        xgb_test.fit(test_data, test_labels)
        print("🚀 GPU support detected and working for XGBoost!")
        return True
    except Exception as e:
        print(f"⚠️ XGBoost GPU not available: {str(e)}")
        print("Using CPU instead")
        return False

gpu_available = check_xgboost_gpu()

# ===== DIFFERENTIAL EVOLUTION OPTIMIZATION =====
print("\n========== Starting Hyperparameter Optimization ==========")

# Global variables for optimization
optimization_iteration = 0
best_score_so_far = float('inf')

def xgboost_objective(params):
    """Objective function for DE optimization - FIXED VERSION"""
    global optimization_iteration, best_score_so_far
    optimization_iteration += 1
    
    # Unpack parameters
    max_depth = int(params[0])
    learning_rate = params[1]
    n_estimators = int(params[2])
    subsample = params[3]
    colsample_bytree = params[4]
    gamma = params[5]
    min_child_weight = params[6]
    reg_alpha = params[7]
    reg_lambda = params[8]
    
    # Create XGBoost model
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=6,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        tree_method='hist',
        device='cuda' if gpu_available else 'cpu',
        verbosity=0
    )
    
    try:
        # Use GroupKFold for sender-aware CV without sample_weight in cross_val_score
        from sklearn.model_selection import GroupKFold
        
        # Create groups based on senders
        groups = X_train_val['senderpseudo'].values
        group_kfold = GroupKFold(n_splits=3)
        
        scores = []
        sample_weights = np.array([class_weights[y] for y in y_train_val])
        
        for train_idx, val_idx in group_kfold.split(X_train_val_scaled, y_train_val, groups=groups):
            X_train_fold = X_train_val_scaled[train_idx]
            X_val_fold = X_train_val_scaled[val_idx]
            y_train_fold = y_train_val.iloc[train_idx]
            y_val_fold = y_train_val.iloc[val_idx]
            weights_fold = sample_weights[train_idx]
            
            # Train with sample weights
            model.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)
            
            # Predict and calculate F1-macro
            y_pred_fold = model.predict(X_val_fold)
            f1 = f1_score(y_val_fold, y_pred_fold, average='macro')
            scores.append(f1)
        
        score = -np.mean(scores)  # Negative because DE minimizes
        
        if score < best_score_so_far:
            best_score_so_far = score
            print(f"Iteration {optimization_iteration}: New best F1-macro: {-score:.4f}")
            print(f"  Params: depth={max_depth}, lr={learning_rate:.3f}, n_est={n_estimators}")
        
        return score
    except Exception as e:
        print(f"Error in iteration {optimization_iteration}: {e}")
        return 1.0  # Return bad score if error

# Parameter bounds for optimization
bounds = [
    (3, 10),      # max_depth
    (0.01, 0.3),  # learning_rate
    (100, 1000),  # n_estimators
    (0.6, 1.0),   # subsample
    (0.6, 1.0),   # colsample_bytree
    (0, 5),       # gamma
    (1, 10),      # min_child_weight
    (0, 10),      # reg_alpha (L1)
    (0, 10),      # reg_lambda (L2)
]

print("Starting Differential Evolution optimization...")
print("This will take approximately 30-60 minutes...")
print("Progress will be shown for improvements only...")

# Run DE optimization
result = differential_evolution(
    xgboost_objective,
    bounds,
    maxiter=20,        # 20 generations
    popsize=10,        # 10 individuals per generation
    mutation=(0.5, 1.5),
    recombination=0.9,
    seed=42,
    disp=True,
    workers=1          # Sequential to avoid memory issues
)

print("\n========== Optimization Completed! ==========")
print(f"Best F1-macro score: {-result.fun:.4f}")
print("Best parameters:")

param_names = ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 
               'colsample_bytree', 'gamma', 'min_child_weight', 'reg_alpha', 'reg_lambda']

best_params = {}
for i, (name, value) in enumerate(zip(param_names, result.x)):
    if name in ['max_depth', 'n_estimators', 'min_child_weight']:
        best_params[name] = int(value)
    else:
        best_params[name] = round(value, 4)
    print(f"{name}: {best_params[name]}")

# ===== TRAIN FINAL OPTIMIZED MODEL =====
print("\n========== Training Final Optimized Model ==========")

# Create optimized model
optimized_params = {
    'objective': 'multi:softmax',
    'num_class': 6,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': -1,
    **best_params
}

# Add GPU parameters if available
if gpu_available:
    optimized_params.update({
        'tree_method': 'hist',
        'device': 'cuda'
    })

print("Training optimized model...")
final_model = XGBClassifier(**optimized_params)
sample_weights = np.array([class_weights[y] for y in y_train_val])

# Train final model
final_model.fit(X_train_val_scaled, y_train_val, sample_weight=sample_weights)

# ===== EVALUATE ON TEST SET =====
print("\n========== Evaluating Optimized Model ==========")

# Scale test data
X_test_scaled = scaler.transform(X_test_unseen)

# Make predictions
y_test_pred = final_model.predict(X_test_scaled)
y_test_proba = final_model.predict_proba(X_test_scaled)

# Calculate metrics
test_accuracy = accuracy_score(y_test_unseen, y_test_pred)
test_f1_macro = f1_score(y_test_unseen, y_test_pred, average='macro')
test_f1_weighted = f1_score(y_test_unseen, y_test_pred, average='weighted')

print(f"========== Optimized XGBoost Performance ==========")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Macro: {test_f1_macro:.4f}")
print(f"Test F1-Weighted: {test_f1_weighted:.4f}")

print("\nOptimized Test Set Classification Report:")
print(classification_report(y_test_unseen, y_test_pred))

# ===== SAVE RESULTS =====
print("\n========== Saving Results ==========")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save optimized model
joblib.dump(final_model, f'{ensemble_dir}/xgboost_optimized_model_{timestamp}.pkl')
joblib.dump(scaler, f'{ensemble_dir}/xgboost_optimized_scaler_{timestamp}.pkl')

# Save optimization results
optimization_results = {
    'best_params': best_params,
    'best_f1_score': -result.fun,
    'test_accuracy': test_accuracy,
    'test_f1_macro': test_f1_macro,
    'test_f1_weighted': test_f1_weighted,
    'optimization_iterations': optimization_iteration,
    'gpu_used': gpu_available,
    'feature_count': X_train_val_scaled.shape[1],
    'training_samples': len(y_train_val),
    'test_samples': len(y_test_unseen),
    'class_weights': class_weights,
    'classification_report': classification_report(y_test_unseen, y_test_pred, output_dict=True)
}

joblib.dump(optimization_results, f'{ensemble_dir}/optimization_results_{timestamp}.pkl')

# Save predictions
predictions_dict = {
    'test_predictions': y_test_pred,
    'test_probabilities': y_test_proba,
    'true_labels': y_test_unseen.values,
    'optimized_params': best_params,
    'timestamp': timestamp
}

joblib.dump(predictions_dict, f'{ensemble_dir}/optimized_predictions_{timestamp}.pkl')

# ===== GENERATE VISUALIZATIONS =====
print("\n========== Generating Visualizations ==========")

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_unseen, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Optimized XGBoost Confusion Matrix\nAccuracy: {test_accuracy:.4f}, F1-Macro: {test_f1_macro:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{ensemble_dir}/optimized_confusion_matrix_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.show()

# Feature Importance
plt.figure(figsize=(12, 8))
feature_importance = final_model.feature_importances_
feature_names = list(X_train_val.columns)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(15)

plt.barh(range(len(importance_df)), importance_df['importance'])
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importance (Optimized XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{ensemble_dir}/optimized_feature_importance_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n🎉 Hyperparameter Optimization Complete!")
print(f"📈 Optimized accuracy: {test_accuracy:.4f}")
print(f"📈 Optimized F1-macro: {test_f1_macro:.4f}")
print(f"🔧 Best parameters found and saved")
print(f"📁 All results saved in: {ensemble_dir}")

# ===== COMPARISON WITH BASELINE =====
baseline_accuracy = 0.8691  # Your previous best result
improvement = test_accuracy - baseline_accuracy

print(f"\n========== Performance Comparison ==========")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"Optimized accuracy: {test_accuracy:.4f}")
print(f"Improvement: {improvement:+.4f} ({improvement/baseline_accuracy*100:+.2f}%)")

if improvement > 0:
    print("✅ Optimization successful - model improved!")
else:
    print("⚠️ No improvement - baseline parameters may already be optimal")

print("\n🚀 Ready for production deployment!")
