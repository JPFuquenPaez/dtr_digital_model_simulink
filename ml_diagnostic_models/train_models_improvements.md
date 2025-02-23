# ML Runs Structure

This directory contains the structure for ML runs in the `dtr_digital_model_simulink` repository.

Run Matlab code to extract training data to .csv.

.mat file consist of 3600 tables, each with 1k rows + 3600 of target data.

Several tries were made to handle .mat data file with python and libraries like scipy without success. Conclusion was that the safest and more efficient way to handle that is within matlab. Hereunder the Matlab code to convert the file to .csv to be later exploited with python. 

>> 
% For training_dataset:

% Load the .mat file
load('training_dataset copy.mat');

% Initialize an empty table to store the combined data
combinedTable = table();

% Loop through each table in varData
for i = 1:length(varData)
    % Get the current table
    currentTable = varData{i};

    % Replicate the target value to match the number of rows in the current table
    targetValues = repmat(y(i), height(currentTable), 1);

    % Add the target values to the current table
    currentTable.Target = targetValues;

    % Concatenate the current table to the combined table
    combinedTable = [combinedTable; currentTable];
end

% Write the combined data to a single CSV file
writetable(combinedTable, 'combined_data.csv');

% For real_testing_dataset:

% Load the .mat file
load('real_testing_dataset copy.mat');

% Initialize an empty table to store the combined data
combinedTable = table();

% Loop through each table in varData
for i = 1:length(dataTables)
    % Get the current table
    currentTable = dataTables{i};

    % Replicate the target value to match the number of rows in the current table
    targetValues = repmat(y(i), height(currentTable), 1);

    % Add the target values to the current table
    currentTable.Target = targetValues;

    % Concatenate the current table to the combined table
    combinedTable = [combinedTable; currentTable];
end

% Write the combined data to a single CSV file
writetable(combinedTable, 'combined_testing_data.csv');
>> 

This generates a .csv file of 1,03 GB. 

# Data shape

3,600,000 rows x 13 columns

Columns 13
Rows 3,600,000

Rows with missing values
0 (0.0%)
Duplicate rows
2,361 (0.1%)

All variables have no missing values and a significant number of distinct values. The histograms suggest that the data for each motor command is fairly evenly distributed, all seem to be normally distributed across their respective ranges (apart from DesiredTrajectory-z, RealizedTrajectory-z with left-skewed distributions).

# LSTM-Based Fault Diagnosis with Digital Twin Features (PyTorch Implementation)

This PyTorch implementation reproduces the fault diagnosis experiment, with critical improvements in loss function implementation and architectural variations with respect to what was done in the matlab implementation found in `docs/trainOnSimulationData.md.`

We demonstrate:

1. Proper handling of CrossEntropyLoss (without redundant softmax)
2. Impact of deeper architectures (3 LSTM layers vs 2)
3. Extended evaluation with ROC curves
4. (Work in progress) Interpretability analysis with integrated gradients using Captum.
5. (Work in progress) Modelling with additional digital twin generate data.
6. (Work in progress) Different architectures in mind (ideas: semisupervised learning, seq2seq + transformers).
7. (Work in progress) Restructure experiments with MLFlow.

**Key Technical Improvements**:
- Removed explicit softmax layer (exploits CrossEntropyLoss's implicit softmax)
- Added gradient clipping (max_norm=1.0) for stable training
- Implemented early stopping with patience=5
- Added comprehensive metrics (AUC-ROC, per-class F1, confusion matrices)

## Dataset Preparation

### Data Loading and Sequencing
```python
# Sequence creation based on timestamp resets
sequences = []
current_sequence = []
current_labels = []
for i in range(len(X_scaled)):
    if i > 0 and timestamps.iloc[i] == 0.0:  # Sequence boundary
        sequences.append((current_sequence, current_labels))
        current_sequence, current_labels = [], []
    current_sequence.append(X_scaled[i])
    current_labels.append(y.iloc[i])
```

## Padding and Splitting

Zero-padded feature vectors

Label propagation: Replicate last valid label for padded positions

Preserves label consistency within sequences

```python
# Uniform sequence length handling
sequence_length = max(len(seq[0]) for seq in sequences)
padding = [[0]*num_features]*(sequence_length - len(seq))
labels.extend([labels[-1]]*(sequence_length - len(labels)))  # Label propagation

# Stratified split preserving class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_padded_flat, 
    test_size=0.2, 
    stratify=y_padded_flat
)
```

## Model Architecture
### Loss Implementation: Critical Fix: Removed explicit softmax layer since CrossEntropyLoss internally applies softmax

PyTorch's CrossEntropyLoss expects raw logits (unnormalized scores)

Explicit softmax + CE loss would double-apply softmax, causing numerical instability ( a test was performed and confirm training instability)

Fix: Remove final softmax layer, use direct linear outputs



# Architectural Variants:
## Base Model (2 LSTM Layers):
nn.LSTM(input_size, hidden_size) → Dropout(0.1) → 
nn.LSTM(hidden_size, hidden_size) → Linear(hidden_size, num_classes)

## Enhanced Model (3 LSTM Layers):
nn.LSTM(input_size, hidden_size) → Dropout → 
nn.LSTM(hidden_size, hidden_size) → Dropout → 
nn.LSTM(hidden_size, hidden_size) → Linear


# Training Curves

## Left: Train/validation loss showing effective convergence



## Right: Gradient norms demonstrating stable optimization


# Confusion Matrix


# ROC Curve