import os
import cv2
from image_processing import func

path = "data/train"
output_path = "data"
train_path = os.path.join(output_path, "train")
test_path = os.path.join(output_path, "test")

# Create output directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

var = 0
c1 = 0
c2 = 0

for subdirectory in os.listdir(path):
    subdirectory_path = os.path.join(path, subdirectory)

    if os.path.isdir(subdirectory_path):
        print("Processing subdirectory:", subdirectory_path)
        
        # Create subdirectories in the output paths
        os.makedirs(os.path.join(train_path, subdirectory), exist_ok=True)
        os.makedirs(os.path.join(test_path, subdirectory), exist_ok=True)

        # List files in the subdirectory
        files = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]

        num = int(0.75 * len(files))

        for i, file in enumerate(files):
            print("Processing file:", file)
            var += 1

            actual_path = os.path.join(subdirectory_path, file)
            actual_path1 = os.path.join(train_path, subdirectory, file)
            actual_path2 = os.path.join(test_path, subdirectory, file)

            img = cv2.imread(actual_path, 0)
            bw_image = func(actual_path)

            if i < num:
                c1 += 1
                cv2.imwrite(actual_path1, bw_image)
            else:
                c2 += 1
                cv2.imwrite(actual_path2, bw_image)

print("Total number of images processed:", var)
print("Count of images in the training set:", c1)
print("Count of images in the testing set:", c2)
