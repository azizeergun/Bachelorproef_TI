import pandas as pd
import re

df = pd.read_csv('deepface_results_black.csv', converters={'Result': lambda x: x})

df[['Verified', 'Distance']] = df['Result'].apply(lambda x: x.split(',')[:2] 
if ':' in x else [None, None]).apply(pd.Series)

def extract_distance(result):
    # Extract the distance value from the result string
    match = re.search(r"'distance': ([\d.]+)", result)
    if match:
        return float(match.group(1))
    return None

def extract_verified(result):
    # Extract the verified value from the result string
    match = re.search(r"'verified': ([\w]+)", result)
    if match:
        return match.group(1)
    return None

df['Verified'] = df['Result'].apply(extract_verified)
df['Distance'] = df['Result'].apply(extract_distance)


verification_counts = df[df['Verified'] == 'True']['Model'].value_counts()
verification_counts.plot.bar(title='Number of times each model correctly verified the images',xlabel='Model', ylabel='Number of verifications')

# To plot the distance for each model:

plt.figure(figsize=(10,6))
plt.title('Distance for each model')

# Set the x and y labels
plt.xlabel('Model')
plt.ylabel('Distance')

# Set the color map to 'plasma'
plt.scatter(df['Model'], df['Distance'], cmap='plasma')

# Show the plot
plt.show()


#compare Black and White
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in both CSV files as dataframes
df1 = pd.read_csv('deepface_results_white_2.csv', converters={'Result': lambda x: x})
df2 = pd.read_csv('deepface_results_black.csv', converters={'Result': lambda x: x})


def extract_verified(result):
    # Extract the verified value from the result string
    match = re.search(r"'verified': ([\w]+)", result)
    if match:
        return match.group(1)
    return None

# Extract the necessary data from the Result column
df1['Verified'] = df1['Result'].apply(extract_verified)
df2['Verified'] = df2['Result'].apply(extract_verified)

# Calculate the verification counts per model in both dataframes
verification_counts1 = df1[df1['Verified'] == 'True']['Model'].value_counts()
verification_counts2 = df2[df2['Verified'] == 'True']['Model'].value_counts()

# Combine the verification counts into a single DataFrame
verification_counts = pd.concat([verification_counts1, verification_counts2], axis=1, sort=True)
verification_counts.columns = ['White', 'Black']

# Plot the verification counts next to each other
verification_counts.plot.barh(title='Number of times each model correctly verified the images', xlabel='Model', ylabel='Number of verifications')
plt.show()



fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first verification count data on the left subplot
verification_counts1.plot.bar(ax=axs[0], color='red', title='Verification counts - CSV 1')

# Plot the second verification count data on the right subplot
verification_counts2.plot.bar(ax=axs[1], color='blue', title='Verification counts - CSV 2')

plt.show()

verification_counts1.plot.bar(color='red', alpha=0.5, label='White')
verification_counts2.plot.bar(color='blue', alpha=0.5, label='Black')
plt.legend()
plt.title('Number of times each model correctly verified the images')
plt.xlabel('Model')
plt.ylabel('Number of verifications')
plt.show()


models = list(set(df1['Model']) | set(df2['Model']))

x_axis = np.arange(len(models))
tick_locations = [value + 0.125 for value in x_axis]
plt.xticks(tick_locations, models)

plt.ylim(0, max(verification_counts1.max(), verification_counts2.max()) + 1)

plt.bar(x_axis, verification_counts1, width=0.25, label='White')
plt.bar(x_axis, verification_counts2, width=0.25, align='edge', label='Black')
plt.legend()

plt.title('Number of times each model correctly verified the images')
plt.xlabel('Model')
plt.ylabel('Number of verifications')
plt.show()

