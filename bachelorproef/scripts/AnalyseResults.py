import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('deepface_results_all_altered.csv', converters={'Result': lambda x: x})

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


verification_counts = df[df['Verified'] == 'False']['Model'].value_counts()
verification_counts.plot.bar(figsize=(10, 6), color= '#ef8667', title='Altered - Number of times each model incorrectly verified the images',xlabel='Model', ylabel='Number of verifications')

for index, value in enumerate(verification_counts):
    label = str(value) # Convert the value to a string
    plt.annotate(label, xy=(index, value), xytext=(0, 3),textcoords="offset points", ha='center', va='bottom')
    
# To plot the distance for each model:

plt.figure(figsize=(10,6))
plt.title('Distance for each model')

# Set the x and y labels
plt.xlabel('Model')
plt.ylabel('Distance')

# Set the color map to 'plasma'
plt.scatter(df['Model'], df['Distance'], color='#ef8667', cmap='plasma')

# Show the plot
plt.show()
