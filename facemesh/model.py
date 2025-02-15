import joblib
import numpy as np
import cv2
import mediapipe as mp
import numpy as np

def generate_face_mesh_array(image_path):
    # Load MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False)

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
    # Flatten it to match SVM input (468 * 3 = 1404)
    landmark_array = landmark_array.flatten().reshape(1, -1)  # Shape: (1, 1404)
    return landmark_array

# Example usage
image_path = "ananya_sad.jpg"  # Change this to your image path
face_mesh_array = generate_face_mesh_array(image_path)

print("Face Mesh Array Shape:", face_mesh_array.shape)
print(face_mesh_array)  # Prints the (x, y, z) coordinates of all 468 landmarks



# Load the trained model and label encoder
model = joblib.load("svm_emotion_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Example new face mesh array (replace with actual data)
#new_face_mesh = np.random.rand(1, X.shape[1])  # Simulate a new face mesh input
new_face_mesh = face_mesh_array
# Predict emotion
predicted_label = model.predict(face_mesh_array)

# Convert back to text
predicted_emotion = label_encoder.inverse_transform(predicted_label)

print(f"Predicted Emotion: {predicted_emotion[0]}")
