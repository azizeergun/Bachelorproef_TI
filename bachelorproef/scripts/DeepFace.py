import itertools
import os
import cv2
import csv
from deepface import DeepFace

files = os.listdir('White')

# Filter out only the image files
image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.webp')]

sorted_image_files = sorted(image_files)

image_names = [f.split('.')[0] for f in sorted_image_files]

# Load the image files into a list of numpy arrays
dataset = []
for file in sorted_image_files:
    img = cv2.imread(os.path.join('White', file))
    dataset.append(img)
    
# Store the image names in a separate list

model = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace','ArcFace','DeepID','Dlib']

# Open the file in write mode
with open('deepface_results_all_white.csv', 'w') as csvfile:
  # Create a CSV writer
  writer = csv.writer(csvfile)
  # Write the headers to the file
  writer.writerow(['Model', 'Image 1', 'Image 2', 'Result'])

for i in range(0, len(dataset) -1, 4):
    print(i)
    for m in model:
        try:
            for x in itertools.combinations(range(0,4),2):
                result = DeepFace.verify(dataset[i + x[0]], dataset[i + x[1]], model_name=m)
                #print(image_names[i + x[0]])
                #print(image_names[i + x[1]])
                with open('deepface_results_all_white.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([m, image_names[i + x[0]], image_names[i + x[1]], result])
        except ValueError as e:
            print(e)
            continue
        except IndexError as e:
            print(e)
            continue
