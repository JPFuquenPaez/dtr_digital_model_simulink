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

# CDAN-Based Cross-Domain Fault Diagnosis with Transformer Features (PyTorch Implementation)

This implementation demonstrates a Conditional Domain Adversarial Network (CDAN) for fault diagnosis in electromechanical systems, bridging the domain gap between simulation data (source domain) and real-world operation data (target domain).

## Key Features
- **Domain Adaptation**: CDAN architecture with gradient reversal and conditional adversarial learning
- **Transformer-Based Feature Extraction**: Learnable positional encodings with residual connections
- **Sequence-Aware Processing**: Temporal feature engineering and sequence modeling
- **Advanced Training**: Mixed precision training, adaptive domain weighting, and early stopping

## Dataset Preparation (3.6M samples, 13 features)

### Feature Engineering
```python
data['Residual-x'] = data['DesiredTrajectory-x'] - data['RealizedTrajectory-x']
# Similar for y and z axes
```

### Sequence Handling
- **Time-based segmentation**: 10s sequences (0.0-9.99 timestamps)
- **Smart Padding**: Zero-padding with label propagation
- **Stratified Splitting**: Preserves class distribution across domains

```python
Sequence length: 1000 timesteps
Train/Val split: 80/20 stratified
Batch size: 32 (both domains)
```

## Model Architecture

### CDAN Components
1. **Feature Extractor** (Transformer Encoder):
   - 6-10 transformer layers with 8-64 attention heads
   - Learnable positional encodings
   - CLS token pooling for sequence aggregation

2. **Label Predictor**:
   - Two-layer MLP with GELU activation
   - Output: 3600 fault classes

3. **Domain Classifier**:
   - Adversarial network with gradient reversal
   - Random matrix approximation for tensor products
   - Adaptive α scheduling (0→1 during training)

![CDAN Architecture](cdan_model_architecture.png)

## Training Process

### Key Mechanisms
```python
# Adaptive domain weighting
alpha = 2. / (1. + np.exp(-5 * (epoch/num_epochs))) - 1

# Mixed precision training
with autocast():
    # Forward passes for both domains
    loss = classification_loss + domain_loss

# Gradient optimization
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Hyperparameters
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (patience=4, factor=0.5)
- **Regularization**: Dropout (0.1-0.2), Layer/Batch Norm
- **Early Stopping**: Patience=15 epochs

## Evaluation Metrics

### Comprehensive Diagnostics
1. **Classification Report**: Precision/Recall/F1 per class
2. **Confusion Matrix**: Heatmap visualization
3. **ROC Analysis**: Multi-class AUC-ROC curves
4. **Domain Alignment**: Loss trajectory analysis


## Technical Improvements

### Critical Implementation Details
1. **Gradient Reversal Layer**:
   ```python
   class GradientReversal(torch.autograd.Function):
       @staticmethod
       def backward(ctx, grad_output):
           return grad_output.neg() * ctx.alpha
   ```

2. **Transformer Enhancements**:
   - Learnable positional encodings
   - Residual connections between encoder layers
   - CLS token aggregation with batch normalization

3. **Domain Adaptation**:
   - Randomized multilinear conditioning
   - Progressive domain weighting (α scheduling)
   - Balanced source/target batch sampling

## Training Dynamics

### Learning Curves
![Learning Curves](training_curves.png)


### Data Preparation
```python
data_info = load_and_preprocess_data(
    source_path='combined_data.csv',
    target_path='combined_testing_data.csv'
)
```

### Model Initialization
```python
model = CDAN(
    input_size=13,
    hidden_size=512,
    num_classes=3600,
    max_len=1000,
    num_heads=16,
    num_layers=10
).to(device)
```

### Training Execution
```python
model, history = train_cdan(
    model=model,
    source_loader=data_info['train_loader'],
    target_loader=data_info['target_loader'],
    num_epochs=150,
    early_stopping_patience=15
)
```


## Next Steps
1. Integrate MLFlow for experiment tracking