# import cv2
# import mediapipe as mp
# import os
# import csv

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# # Paths
# input_folder = "dataset/sad_resized_images"    # Folder with resized images
# csv_file = "face_mesh_data.csv"    # Output CSV file

# # Open CSV file to store face mesh data
# with open(csv_file, "w", newline="") as file:
#     writer = csv.writer(file)
    
#     # Write header (Image Name + 468 landmark points)
#     header = ["Image"]
#     for i in range(468):
#         header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
#     header.append("Emotion")  # Emotion label (e.g., Happy, Sad)
    
#     writer.writerow(header)

#     # Process each image
#     for filename in os.listdir(input_folder):
#         img_path = os.path.join(input_folder, filename)
#         img = cv2.imread(img_path)

#         if img is not None:
#             # Convert image to RGB for MediaPipe
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#             # Detect face landmarks
#             results = face_mesh.process(img_rgb)

#             if results.multi_face_landmarks:
#                 for face_landmarks in results.multi_face_landmarks:
#                     row = [filename]  # Start with image name
                    
#                     # Extract 468 (x, y, z) coordinates
#                     for landmark in face_landmarks.landmark:
#                         row.extend([landmark.x, landmark.y, landmark.z])

#                     row.append("Sad")  # Assign emotion label

#                     # Write to CSV
#                     writer.writerow(row)
#                     print(f"Processed: {filename}")
#         else:
#             print(f"Error loading image: {filename}")

# print("✅ Face mesh data saved to face_mesh_data.csv")

import cv2
import mediapipe as mp
import os
import csv

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Paths
input_folder = "dataset/sad_resized_images"  # Folder with resized images
csv_file = "face_mesh_data.csv"  # Output CSV file

# Open CSV file in append mode (so it doesn't overwrite existing data)
with open(csv_file, "a", newline="") as file:  # ✅ Changed "w" to "a"
    writer = csv.writer(file)

    # Process each image
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            # Convert image to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect face landmarks
            results = face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    row = [filename]  # Start with image name
                    
                    # Extract 468 (x, y, z) coordinates
                    for landmark in face_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z])

                    row.append("Sad")  # Assign emotion label

                    # Write to CSV (append mode)
                    writer.writerow(row)
                    print(f"Processed: {filename}")
        else:
            print(f"Error loading image: {filename}")

print("✅ Face mesh data appended to face_mesh_data.csv")
