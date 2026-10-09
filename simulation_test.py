from datasets import load_dataset
import numpy as np

# Define the data set 
dataset = load_dataset("inria-soda/tabular-benchmark", data_files="reg_cat/diamonds.csv")
df = dataset["train"].to_pandas()

# Generate missing values based on an MNAR missingness p = (-1, -0.8, 0, 0.8, 1)
df_miss = df.copy()
mask = np.random.rand(*df_miss.shape) < 0.20 # missingness level is 20% 
df_miss = df_miss.mask(mask)

print(df_miss.isna().mean().mean())

# Impute this data set for each p 
rho=0.5 


# Store the imputed value for each observation


# Calculate the variance of the imputed values for each observation


# Use Rubin's rules to pool the imputed values and create a complete data set 


# Use a standard noise corruption process 


# Add weighting to the loss function proportional to variance of the imputed values


# Then optimize the DAE parameters using the weighted loss function





