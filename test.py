import tensorflow as tf
import numpy as np
import argparse

# Parse command-line arguments for flexibility
parser = argparse.ArgumentParser(description="Evaluate a trained model on test data.")
parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file.")
parser.add_argument('--window', type=int, default=10, help="Window size for data reshaping.")
parser.add_argument('--step', type=int, default=1, help="Step size for data processing.")
parser.add_argument('--data_test', type=str, required=True, help="Path to the test data file (e.g., .npy).")
args = parser.parse_args()

# Load the test data
try:
    data_test = np.load(args.data_test)
except Exception as e:
    print(f"Error loading test data: {e}")
    raise

# Load the pre-trained model
try:
    modelt = tf.keras.models.load_model(args.model_path, compile=False)
    print("Model loaded successfully with Keras (compile=False)")
except Exception as e:
    print(f"Error loading model from {args.model_path}: {e}")
    raise

# Prepare the test data
window = args.window
step = args.step
data_test_for_evaluate = data_test[:, 1:].reshape((len(data_test) // window, window, 1))
targets_test = data_test[:, :1].reshape((len(data_test) // window, window, 1))

# Initialize arrays for predictions and true targets
sample = np.zeros((1, window // step, data_test.shape[-1] - 1))
predicted_targets = np.zeros((len(data_test_for_evaluate),))
true_targets = np.zeros((len(data_test_for_evaluate),))

# Compute the true targets
for i in range(len(data_test_for_evaluate)):
    true_targets[i] = targets_test[i, window - 1]
target_mean = true_targets.mean(axis=0)

# Predict using the loaded model
for i in range(len(data_test_for_evaluate)):
    sample[0] = data_test_for_evaluate[i]
    predicted_targets[i] = modelt.predict(sample)

# Calculate error metrics
MSE = np.mean((predicted_targets - true_targets) ** 2)
MAE = np.mean(np.abs(predicted_targets - true_targets))
RRSE = 100 * np.sqrt(MSE * len(true_targets) / np.sum((true_targets - target_mean) ** 2))
RAE = 100 * MAE * len(true_targets) / np.sum(np.abs(true_targets - target_mean))

# Print evaluation metrics
print('MSE:', MSE)
print('MAE:', MAE)
print('RRSE:', RRSE)
print('RAE:', RAE)
print('target_mean:', target_mean)
print('len(true_targets):', len(true_targets))
