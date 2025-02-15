import cv2
import mediapipe as mp
import numpy as np

def generate_face_mesh_array(image_path):
    # Load MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not load the image. Check the path.")

    # Convert BGR to RGB (MediaPipe requires RGB input)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        raise ValueError("No face detected in the image.")

    # Extract face mesh landmarks
    face_landmarks = results.multi_face_landmarks[0]  # Get the first detected face
    landmark_array = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])

    return landmark_array

# Example usage
image_path = "shiv.jpg"  # Change this to your image path
face_mesh_array = generate_face_mesh_array(image_path)

print("Face Mesh Array Shape:", face_mesh_array.shape)
print(face_mesh_array)  # Prints the (x, y, z) coordinates of all 468 landmarks
