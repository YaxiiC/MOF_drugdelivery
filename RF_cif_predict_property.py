import os
import numpy as np
import pandas as pd

from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================
# Custom CIF Featurizer
# ==========================
class CustomCIFFeaturizer:
    def featurize(self, cif_file_path):
        try:
            # Parse the structure from CIF file
            parser = CifParser(cif_file_path, occupancy_tolerance=1.1)
            structure = parser.get_structures()[0]

            # Basic atomic features: positions and lattice parameters
            atomic_positions = structure.cart_coords.flatten()
            lattice_params = structure.lattice.parameters
            
            # Additional atomic features
            atomic_masses = np.array([site.specie.atomic_mass for site in structure.sites], dtype=float)
            electronegativities = np.array([Element(site.specie.symbol).X for site in structure.sites], dtype=float)
            atomic_numbers = np.array([site.specie.Z for site in structure.sites], dtype=float)
            
            # Structure-wide features
            unit_cell_volume = structure.lattice.volume
            density = structure.density
            avg_atomic_mass = np.mean(atomic_masses)
            total_atomic_number = np.sum(atomic_numbers)
            
            # Coordination numbers for each atom
            coordination_numbers = np.array(
                [len(structure.get_neighbors(site, 3.0)) for site in structure.sites],
                dtype=float
            )
            total_coordination = np.sum(coordination_numbers)
            avg_coordination = np.mean(coordination_numbers)

            # Combine all features into one vector
            feature_vector = np.concatenate([
                atomic_positions,        # Atomic positions
                lattice_params,          # Lattice parameters
                atomic_masses,           # Atomic masses
                electronegativities,     # Electronegativity
                atomic_numbers,          # Atomic numbers               # Van der Waals radii
                [unit_cell_volume],      # Unit cell volume
                [density],               # Density
                [avg_atomic_mass],       # Average atomic mass
                [total_atomic_number],   # Total atomic number

            
                [total_coordination],    # Total coordination number
                [avg_coordination],      # Average coordination number
     
                [std_bond_length]        # Standard deviation of bond lengths
            ])
            
            return feature_vector

        except Exception as e:
            print(f"Failed to featurize {cif_file_path}: {e}")
            return None

# ==========================
# Data Loading & Featurization
# ==========================
print("Script started successfully.")

# Load the properties CSV
properties_path = '/Users/chrissychen/Documents/PhD_2nd_year/MOF_MATRIX/zeyi_msc_project/Filtered_Dataset.csv'
properties_df = pd.read_csv(properties_path)
print(f"Loaded properties from {properties_path}. Shape: {properties_df.shape}")

# Use first 6000 samples (as in your original script)
mof_files = properties_df['Filename'].values[:6000]

# Target column
target_column = 'Included Sphere Along Free Sphere Path'
target_values = properties_df[target_column].values[:6000]

# Directory with CIF files
cif_directory = '/Users/chrissychen/Documents/PhD_2nd_year/MOF_MATRIX/2022_CSD_MOF_Collection'

# Initialize featurizer
featurizer = CustomCIFFeaturizer()
print("Featurizer initialized.")

features = []
labels = []
skipped_files = []

for mof_file, target in zip(mof_files, target_values):
    cif_path = os.path.join(cif_directory, f"{mof_file}.cif")
    print(f"Featurizing {cif_path}...")
    feature = featurizer.featurize(cif_path)

    if feature is not None:
        features.append(feature)
        labels.append(target)
    else:
        print(f"Failed to featurize {cif_path}. Skipping.")
        skipped_files.append(cif_path)

# ==========================
# Padding / Truncation of Features
# ==========================
print("Standardizing feature vector lengths (padding/truncation)...")

max_length = max(len(f) for f in features)
padded_features = []

for feature in features:
    if len(feature) < max_length:
        feature = np.pad(feature, (0, max_length - len(feature)), 'constant')
    elif len(feature) > max_length:
        feature = feature[:max_length]
    padded_features.append(feature)

features = np.array(padded_features)
labels = np.array(labels)  # shape (N,)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Save features & labels
np.savez('mof_features_and_labels_random_forest.npz', features=features, labels=labels)
print("Features and labels saved to mof_features_and_labels_random_forest.npz")

with open('skipped_files_random_forest.log', 'w') as f:
    for item in skipped_files:
        f.write("%s\n" % item)
print(f"Skipped files saved to skipped_files_random_forest.log")

# ==========================
# Random Forest Training & Evaluation (5 Seeds)
# ==========================

seeds = [0, 1, 2, 3, 4]
mae_list = []
mse_list = []
rmse_list = []
r2_list = []

for seed in seeds:
    print(f"\n=== Random Seed: {seed} ===")

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=seed
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

    print(f"Seed {seed} - MAE  = {mae:.4f}")
    print(f"Seed {seed} - MSE  = {mse:.4f}")
    print(f"Seed {seed} - RMSE = {rmse:.4f}")
    print(f"Seed {seed} - R²   = {r2:.4f}")

# ==========================
# Print Mean ± Std over 5 runs
# ==========================
def mean_std_str(values):
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

print("\n================ Overall Performance (5 random seeds) ================")
print(f"MAE  : {mean_std_str(mae_list)}")
print(f"MSE  : {mean_std_str(mse_list)}")
print(f"RMSE : {mean_std_str(rmse_list)}")
print(f"R²   : {mean_std_str(r2_list)}")