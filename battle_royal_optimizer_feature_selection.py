import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import time
import warnings
import os
from tqdm import tqdm
import random

# Import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    print("LightGBM detected for classifier!")
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not found. Please install with: pip install lightgbm")

# Check for GPU-enabled LightGBM
HAS_LIGHTGBM_GPU = False
if HAS_LIGHTGBM:
    try:
        X_test = np.random.random((10, 10))
        y_test = np.random.randint(0, 2, 10)
        test_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
        lgb_test = lgb.LGBMClassifier(**test_params)
        lgb_test.fit(X_test, y_test)
        HAS_LIGHTGBM_GPU = True
        print("GPU support for LightGBM detected! Will use GPU acceleration.")
    except Exception as e:
        print("LightGBM GPU support not available. Using CPU version.")
        HAS_LIGHTGBM_GPU = False

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

def calculate_class_weights(y, method='sqrt'):
    """Calculate class weights for imbalanced datasets"""
    if method is None:
        return None
    classes = np.unique(y)
    if method == 'sqrt':
        class_counts = np.bincount(y)
        total_samples = len(y)
        weights = {cls: np.sqrt(total_samples / (len(classes) * class_counts[cls])) for cls in classes}
        print(f"Class weights (sqrt method): {weights}")
        return weights
    elif method == 'balanced':
        weights = compute_class_weight('balanced', classes=classes, y=y)
        weight_dict = dict(zip(classes, weights))
        print(f"Class weights (balanced method): {weight_dict}")
        return weight_dict
    else:
        print(f"Unknown class weight method: {method}. Using no weights.")
        return None

def filter_classes(X, y, target_classes=[0, 1, 3, 4]):
    """Filter data to only include specified classes and remap class labels"""
    mask = np.isin(y, target_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(sorted(target_classes))}
    y_remapped = np.array([class_mapping[label] for label in y_filtered])
    print(f"Original classes: {sorted(np.unique(y))}")
    print(f"Filtered to classes: {sorted(target_classes)}")
    print(f"Class mapping: {class_mapping}")
    print(f"New class labels: {sorted(np.unique(y_remapped))}")
    print(f"Data shape before filtering: {X.shape}")
    print(f"Data shape after filtering: {X_filtered.shape}")
    print(f"Samples removed: {X.shape[0] - X_filtered.shape[0]} ({((X.shape[0] - X_filtered.shape[0])/X.shape[0]*100):.1f}%)")
    return X_filtered, y_remapped, class_mapping

class Player:
    def __init__(self, n_features, min_features=3, max_features=None):
        self.n_features = n_features
        # Ensure min and max features are within valid range
        self.min_features = min(min_features, n_features)
        self.max_features = min(max_features, n_features) if max_features else n_features // 2
        self.max_features = max(self.min_features, self.max_features)
        # Initialize with random feature subset
        num_features = np.random.randint(self.min_features, self.max_features + 1)
        self.features = np.sort(np.random.choice(range(n_features), size=num_features, replace=False))
        self.fitness = -np.inf

    def mutate(self, mutation_rate=0.1):
        """Battle Royale-specific mutation operator"""
        mutated = self.features.copy()
        # Randomly add features
        if np.random.rand() < mutation_rate and len(mutated) < self.max_features:
            available = np.setdiff1d(range(self.n_features), mutated)
            if len(available) > 0:
                mutated = np.append(mutated, np.random.choice(available))
        # Randomly remove features
        if np.random.rand() < mutation_rate and len(mutated) > self.min_features:
            mutated = np.delete(mutated, np.random.randint(0, len(mutated)))
        return np.sort(np.unique(mutated))

class BattleRoyaleOptimizer:
    def __init__(self, n_players=32, max_generations=30, min_features=3, max_features=None,
                 mutation_rate=0.15, classifier_type='lightgbm', sampling_rate=1.0,
                 use_gpu=True, use_class_weights=True, class_weight_method='sqrt'):
        self.n_players = n_players
        self.max_generations = max_generations
        self.min_features = min_features
        self.max_features = max_features
        self.mutation_rate = mutation_rate
        self.classifier_type = classifier_type.lower()
        self.sampling_rate = sampling_rate
        self.use_gpu = use_gpu
        self.use_class_weights = use_class_weights
        self.class_weight_method = class_weight_method
        self.classifier = None
        self._setup_classifier()
        self.best_solution = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.feature_count_history = []
        self.performance_history = []

    def _setup_classifier(self, force_new=False):
        """Configure the appropriate classifier"""
        if self.classifier is not None and not force_new:
            return self.classifier
        if HAS_LIGHTGBM:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 15,
                'num_leaves': 31,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
            if HAS_LIGHTGBM_GPU and self.use_gpu:
                params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                })
                print("Using LightGBM with GPU acceleration")
            else:
                print("Using LightGBM with CPU")
            self.classifier = lgb.LGBMClassifier(**params)
        else:
            print("Warning: LightGBM not available. Falling back to Random Forest.")
            self.classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
            )
            print("Using Random Forest classifier with CPU")
        return self.classifier

    def _sample_data(self, X, y):
        """Sample data at the specified rate if needed"""
        if self.sampling_rate >= 1.0:
            return X, y
        # Stratified sampling to maintain class distribution
        n_samples = int(X.shape[0] * self.sampling_rate)
        indices = []
        classes = np.unique(y)
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            n_cls_samples = int(len(cls_indices) * self.sampling_rate)
            if n_cls_samples > 0:
                sampled_indices = np.random.choice(cls_indices, size=n_cls_samples, replace=False)
                indices.extend(sampled_indices)
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def _fitness_function(self, X_train, y_train, X_val, y_val, selected_features):
        """Evaluate fitness of a feature subset with class weights"""
        if len(selected_features) < self.min_features:
            return -np.inf, 0, 0, len(selected_features), 0
        X_train_selected = X_train[:, selected_features]
        X_val_selected = X_val[:, selected_features]
        try:
            start_time = time.time()
            # Calculate class weights if enabled
            class_weights = None
            if self.use_class_weights:
                class_weights = calculate_class_weights(y_train, method=self.class_weight_method)
            # Create a fresh classifier instance for this evaluation
            classifier = self._setup_classifier(force_new=True)
            # Set class weights if using LightGBM
            if HAS_LIGHTGBM and isinstance(classifier, lgb.LGBMClassifier) and class_weights:
                classifier.set_params(class_weight=class_weights)
            elif hasattr(classifier, 'class_weight') and class_weights:
                classifier.set_params(class_weight=class_weights)
            classifier.fit(X_train_selected, y_train)
            y_pred = classifier.predict(X_val_selected)
            training_time = time.time() - start_time
            # Calculate primary performance metric
            f1 = f1_score(y_val, y_pred, average='macro')
            # Calculate secondary performance metric (accuracy)
            acc = accuracy_score(y_val, y_pred)
            # Penalty for number of features
            feature_penalty = len(selected_features) / X_train.shape[1]
            # Final fitness: prioritize F1 score
            alpha = 0.9
            beta = 0.1
            fitness = alpha * f1 - beta * feature_penalty
            return fitness, f1, acc, len(selected_features), training_time
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return -np.inf, 0, 0, len(selected_features), 0

    def _compete(self, player1, player2, X_train, y_train, X_val, y_val):
        """Pairwise competition between two players"""
        p1_fitness = self._fitness_function(X_train, y_train, X_val, y_val, player1.features)[0]
        p2_fitness = self._fitness_function(X_train, y_train, X_val, y_val, player2.features)[0]
        if p1_fitness > p2_fitness:
            return player1, player2
        else:
            return player2, player1

    def optimize(self, X_train, y_train, X_val, y_val):
        """Run BRO to find optimal feature subset"""
        n_features = X_train.shape[1]
        min_feats = min(self.min_features, n_features)
        max_feats = min(self.max_features, n_features) if self.max_features else n_features // 2
        max_feats = max(min_feats, max_feats)
        self.min_features, self.max_features = min_feats, max_feats
        # Sample data if sampling rate < 1.0
        if self.sampling_rate < 1.0:
            X_train_sample, y_train_sample = self._sample_data(X_train, y_train)
            print(f"Sampled training data: {X_train_sample.shape[0]} rows ({self.sampling_rate*100:.1f}% of original {X_train.shape[0]} rows)")
        else:
            X_train_sample, y_train_sample = X_train, y_train
            print(f"Using full training dataset: {X_train.shape[0]} rows")
        print("\nClass distribution in training data used for optimization:")
        for cls, count in sorted(zip(*np.unique(y_train_sample, return_counts=True))):
            print(f" Class {cls}: {count} samples ({count/len(y_train_sample)*100:.2f}%)")
        if self.use_class_weights:
            print(f"\nUsing class weights with method: {self.class_weight_method}")
            sample_weights = calculate_class_weights(y_train_sample, method=self.class_weight_method)
        else:
            print("\nNot using class weights")
        # Initialize population
        population = [Player(n_features, self.min_features, self.max_features) for _ in range(self.n_players)]
        # Store progress metrics
        best_fitness_history = []
        avg_fitness_history = []
        feature_count_history = []
        print(f"\nBattle Royale Optimization starting with {self.n_players} players over {self.max_generations} generations")
        print(f"Feature constraints: min={self.min_features}, max={self.max_features}")
        total_start_time = time.time()
        # Main BRO loop
        for generation in range(self.max_generations):
            generation_start_time = time.time()
            # Evaluate all players
            fitness_scores = []
            for player in tqdm(population, desc="Evaluating players"):
                result = self._fitness_function(X_train_sample, y_train_sample, X_val, y_val, player.features)
                player.fitness = result[0]
                fitness_scores.append(player.fitness)
                # Update global best
                if player.fitness > self.best_fitness:
                    self.best_fitness = player.fitness
                    self.best_solution = player.features.copy()
                    self.performance_history.append((generation, result[1], result[2], result[3]))
            # Battle phase
            new_population = []
            random.shuffle(population)
            # Pairwise competitions
            for i in range(0, len(population), 2):
                if i+1 >= len(population):
                    # Handle odd number of players
                    new_population.append(population[i])
                    continue
                # Compete two players
                winner, loser = self._compete(population[i], population[i+1], X_train_sample, y_train_sample, X_val, y_val)
                # Generate offspring through mutation
                offspring_features = winner.mutate(self.mutation_rate)
                offspring = Player(n_features, self.min_features, self.max_features)
                offspring.features = offspring_features
                new_population.extend([winner, offspring])
            # Maintain population size
            population = new_population[:self.n_players]
            # Record progress
            best_fitness_history.append(self.best_fitness)
            avg_fitness_history.append(np.mean(fitness_scores))
            if self.best_solution is not None:
                feature_count_history.append(len(self.best_solution))
            # Print generation summary
            generation_time = time.time() - generation_start_time
            elapsed_time = time.time() - total_start_time
            remaining_time = (elapsed_time / (generation + 1)) * (self.max_generations - generation - 1)
            print(f"\nGeneration {generation+1} completed in {generation_time:.2f}s")
            print(f"Average fitness: {np.mean(fitness_scores):.4f}")
            print(f"Best fitness: {self.best_fitness:.4f}")
            if self.best_solution is not None:
                print(f"Best solution has {len(self.best_solution)} features")
            print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
            print(f"Estimated remaining time: {remaining_time/60:.1f} minutes")
            # Save intermediate results
            if (generation + 1) % 5 == 0 or generation == self.max_generations - 1:
                self._save_intermediate_results(generation, best_fitness_history, avg_fitness_history)
        # Store final results
        self.fitness_history = {
            'best': best_fitness_history,
            'avg': avg_fitness_history
        }
        self.feature_count_history = feature_count_history
        total_time = time.time() - total_start_time
        print(f"\n✅ Battle Royale optimization completed in {total_time/60:.1f} minutes")
        print(f"Best solution found at generation {generation + 1}")
        return self.best_solution

    def _save_intermediate_results(self, generation, best_fitness, avg_fitness):
        """Save intermediate results to file"""
        if self.best_solution is None:
            return
        # Create results directory if it doesn't exist
        if not os.path.exists('bro_results'):
            os.makedirs('bro_results')
        # Save best feature set so far
        np.savetxt(f'bro_results/best_features_gen_{generation+1}.csv', self.best_solution, delimiter=',', fmt='%d')
        # Save fitness progress
        pd.DataFrame({
            'generation': range(len(best_fitness)),
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness
        }).to_csv(f'bro_results/fitness_progress_gen_{generation+1}.csv', index=False)
        # Save quick summary
        with open(f'bro_results/summary_gen_{generation+1}.txt', 'w') as f:
            f.write(f"BRO Feature Selection - Generation {generation+1} Summary\n")
            f.write(f"Best fitness: {self.best_fitness:.4f}\n")
            f.write(f"Number of features: {len(self.best_solution)}\n")
            f.write(f"Selected features: {sorted(self.best_solution)}\n")
            f.write(f"Class weights: {'Enabled (' + self.class_weight_method + ')' if self.use_class_weights else 'Disabled'}\n")

    def plot_progress(self):
        """Plot the optimization progress"""
        plt.figure(figsize=(15, 10))
        # Plot fitness progress
        plt.subplot(2, 1, 1)
        plt.plot(self.fitness_history['best'], 'b-', label='Best Fitness')
        plt.plot(self.fitness_history['avg'], 'r-', label='Average Fitness')
        plt.title('Fitness Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        # Plot feature count
        plt.subplot(2, 1, 2)
        plt.plot(self.feature_count_history, 'g-')
        plt.title('Number of Selected Features')
        plt.xlabel('Generation')
        plt.ylabel('Number of Features')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('bro_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Plot performance metrics from the best solutions found
        if hasattr(self, 'performance_history') and self.performance_history:
            generations, f1_scores, accuracies, feature_counts = zip(*self.performance_history)
            plt.figure(figsize=(15, 10))
            plt.subplot(3, 1, 1)
            plt.plot(generations, f1_scores, 'bo-')
            plt.title('F1 Score Progress (Best Solutions)')
            plt.xlabel('Generation Found')
            plt.ylabel('F1 Score')
            plt.grid(True)
            plt.subplot(3, 1, 2)
            plt.plot(generations, accuracies, 'ro-')
            plt.title('Accuracy Progress (Best Solutions)')
            plt.xlabel('Generation Found')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.subplot(3, 1, 3)
            plt.plot(generations, feature_counts, 'go-')
            plt.title('Feature Count Progress (Best Solutions)')
            plt.xlabel('Generation Found')
            plt.ylabel('Number of Features')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('bro_performance.png', dpi=300, bbox_inches='tight')
            plt.close()

    def evaluate_final_model(self, X_train, y_train, X_test, y_test):
        """Evaluate the final model with the best feature subset"""
        if self.best_solution is None:
            print("No optimization performed yet.")
            return
        selected_features = self.best_solution.astype(int)
        print(f"\nFinal Evaluation with {len(selected_features)} selected features")
        print(f"Selected features indices: {selected_features}")
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        class_weights = None
        if self.use_class_weights:
            class_weights = calculate_class_weights(y_train, method=self.class_weight_method)
            print(f"Using class weights for final model: {class_weights}")
        final_classifier = self._setup_classifier(force_new=True)
        if HAS_LIGHTGBM and isinstance(final_classifier, lgb.LGBMClassifier) and class_weights:
            final_classifier.set_params(class_weight=class_weights)
        elif hasattr(final_classifier, 'class_weight') and class_weights:
            final_classifier.set_params(class_weight=class_weights)
        final_classifier.fit(X_train_selected, y_train)
        y_pred = final_classifier.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        return {
            'selected_features': selected_features,
            'accuracy': accuracy,
            'f1_score': f1,
            'class_weights': class_weights
        }

# Main execution block
if __name__ == "__main__":
    print("\n===== LOADING DATA =====")
    try:
        X_train = pd.read_csv('X_train.csv').values
        y_train = pd.read_csv('y_train.csv').values.ravel()
        X_val = pd.read_csv('X_val.csv').values
        y_val = pd.read_csv('y_val.csv').values.ravel()
        X_test = pd.read_csv('X_test.csv').values
        y_test = pd.read_csv('y_test.csv').values.ravel()
        print(f"✅ Data loaded successfully")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print("\nOriginal class distribution in training set:")
        for cls, count in sorted(zip(*np.unique(y_train, return_counts=True))):
            print(f" Class {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please ensure all CSV files are in the current directory.")
        exit(1)
    print("\n===== FILTERING CLASSES =====")
    target_classes = [0, 1, 3, 4]
    print("\nFiltering training data...")
    X_train_filtered, y_train_filtered, train_class_mapping = filter_classes(X_train, y_train, target_classes)
    print("\nFiltering validation data...")
    X_val_filtered, y_val_filtered, val_class_mapping = filter_classes(X_val, y_val, target_classes)
    print("\nFiltering test data...")
    X_test_filtered, y_test_filtered, test_class_mapping = filter_classes(X_test, y_test, target_classes)
    print("\nFiltered class distribution in training set:")
    for cls, count in sorted(zip(*np.unique(y_train_filtered, return_counts=True))):
        print(f" Class {cls}: {count} samples ({count/len(y_train_filtered)*100:.2f}%)")
    print("\n===== CONFIGURING BATTLE ROYALE OPTIMIZER =====")
    n_features = X_train_filtered.shape[1]
    min_feats = 3
    max_feats = 10
    bro = BattleRoyaleOptimizer(
        n_players=32,
        max_generations=30,
        min_features=min_feats,
        max_features=max_feats,
        mutation_rate=0.15,
        classifier_type='lightgbm',
        sampling_rate=0.2,
        use_gpu=True,
        use_class_weights=True,
        class_weight_method='sqrt'
    )
    try:
        feature_names = pd.read_csv('X_train.csv').columns.tolist()
        print(f"Found {len(feature_names)} feature names from CSV headers")
    except:
        feature_names = [f"Feature_{i}" for i in range(X_train_filtered.shape[1])]
        print(f"Using generic feature names: Feature_0 to Feature_{X_train_filtered.shape[1]-1}")
    print("\n===== STARTING BATTLE ROYALE OPTIMIZATION =====")
    best_features = bro.optimize(X_train_filtered, y_train_filtered, X_val_filtered, y_val_filtered)
    print("\n===== GENERATING VISUALIZATIONS =====")
    bro.plot_progress()
    print("\n===== EVALUATING FINAL MODEL ON FULL FILTERED DATASET =====")
    if best_features is not None:
        selected_features = best_features.astype(int)
        X_train_selected = X_train_filtered[:, selected_features]
        X_test_selected = X_test_filtered[:, selected_features]
        class_weights = None
        if bro.use_class_weights:
            class_weights = calculate_class_weights(y_train_filtered, method=bro.class_weight_method)
            print(f"Final model class weights: {class_weights}")
        final_model = bro._setup_classifier(force_new=True)
        if HAS_LIGHTGBM and isinstance(final_model, lgb.LGBMClassifier) and class_weights:
            final_model.set_params(class_weight=class_weights)
        elif hasattr(final_model, 'class_weight') and class_weights:
            final_model.set_params(class_weight=class_weights)
        final_model.fit(X_train_selected, y_train_filtered)
        y_pred = final_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test_filtered, y_pred)
        f1 = f1_score(y_test_filtered, y_pred, average='macro')
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"Final Test F1 Score: {f1:.4f}")
        results = {
            'selected_features': selected_features,
            'accuracy': accuracy,
            'f1_score': f1,
            'class_mapping': train_class_mapping,
            'class_weights': class_weights
        }
        np.savetxt('bro_selected_features_indices.csv', best_features, delimiter=',', fmt='%d')
        selected_names = [feature_names[i] for i in best_features]
        with open('bro_selected_features_names.txt', 'w') as f:
            f.write("Selected Features for Filtered Classes [0, 1, 3, 4] - BRO\n")
            f.write(f"Class mapping: {train_class_mapping}\n")
            f.write(f"Class weights method: {bro.class_weight_method if bro.use_class_weights else 'None'}\n")
            if class_weights:
                f.write(f"Class weights: {class_weights}\n")
            f.write("\nSelected Features:\n")
            for i, name in zip(best_features, selected_names):
                f.write(f"{i}: {name}\n")
        with open('bro_class_mapping_and_weights.txt', 'w') as f:
            f.write("Class Mapping and Weights Information - BRO\n")
            f.write("==========================================\n")
            f.write(f"Original classes kept: {target_classes}\n")
            f.write(f"Class mapping (original -> new): {train_class_mapping}\n")
            f.write(f"Class weights enabled: {bro.use_class_weights}\n")
            f.write(f"Class weights method: {bro.class_weight_method if bro.use_class_weights else 'None'}\n")
            if class_weights:
                f.write(f"Final class weights: {class_weights}\n")
            f.write(f"Population size: {bro.n_players}\n")
            f.write(f"Max generations: {bro.max_generations}\n")
            f.write(f"Mutation rate: {bro.mutation_rate}\n")
            f.write(f"Total samples before filtering: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}\n")
            f.write(f"Total samples after filtering: {X_train_filtered.shape[0] + X_val_filtered.shape[0] + X_test_filtered.shape[0]}\n")
            f.write(f"Final test F1-score: {f1:.4f}\n")
            f.write(f"Final test accuracy: {accuracy:.4f}\n")
        print(f"\n✅ Selected {len(best_features)} features saved to files:")
        print(f" - bro_selected_features_indices.csv")
        print(f" - bro_selected_features_names.txt")
        print(f" - bro_class_mapping_and_weights.txt")
        print(f" - bro_progress.png")
        print(f" - bro_performance.png")
        print("\nSelected features:")
        for i, feature_idx in enumerate(sorted(best_features)):
            print(f" {i+1}. {feature_names[feature_idx]} (index: {feature_idx})")
        print("\n===== BATTLE ROYALE OPTIMIZATION COMPLETE =====")
        print(f"Final F1-score on filtered test set: {results['f1_score']:.4f}")
        print(f"Final accuracy on filtered test set: {results['accuracy']:.4f}")
        print(f"Number of features reduced from {X_train_filtered.shape[1]} to {len(best_features)}")
        print(f"Feature reduction: {(1 - len(best_features)/X_train_filtered.shape[1])*100:.1f}%")
        print(f"Classes used: {target_classes} (mapped to: {list(train_class_mapping.values())})")
        print(f"Class weights: {'Enabled (' + bro.class_weight_method + ')' if bro.use_class_weights else 'Disabled'}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test_filtered, y_pred, 
                                  target_names=[f"Class_{i}" for i in sorted(np.unique(y_test_filtered))]))
    else:
        print("❌ No best features found during optimization")

