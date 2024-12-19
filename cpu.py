import pandas as pd
import numpy as np

# Number of samples to generate
num_samples = 500

# Generating random data for CPU features
np.random.seed(0)
clock_speed = np.random.uniform(1.0, 5.0, num_samples)  # Clock Speed in GHz
cache_size = np.random.uniform(1.0, 32.0, num_samples)  # Cache Size in MB
power_consumption = np.random.uniform(10.0, 150.0, num_samples)  # Power Consumption in Watts

# Generating a simple performance class based on features (example rules)
performance_class = []
for i in range(num_samples):
    if clock_speed[i] > 3.5 and cache_size[i] > 8:
        performance_class.append("High")
    elif clock_speed[i] > 2.5 and cache_size[i] > 4:
        performance_class.append("Medium")
    else:
        performance_class.append("Low")

# Creating a DataFrame
data = pd.DataFrame({
    'Clock Speed (GHz)': clock_speed,
    'Cache Size (MB)': cache_size,
    'Power Consumption (W)': power_consumption,
    'Performance Class': performance_class
})

# Saving the DataFrame to CSV
data.to_csv('cpu_performance_dataset.csv', index=False)

print("Dataset saved as 'cpu_performance_dataset.csv'")
