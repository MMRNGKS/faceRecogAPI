from flask import Flask, request, jsonify
import face_recognition
from test import test
import imghdr
import firebase_admin
from firebase_admin import credentials, firestore
import requests
from io import BytesIO

app = Flask(__name__)

cred = credentials.Certificate("./thesis-project-9afae-firebase-adminsdk-ruq7x-66ca1f4b45.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Set the tolerance level for face comparison
tolerance = 0.6


def is_valid_image(file):
    """
    Check if the provided file is a valid image.
    """
    image_data = file.read()
    file.seek(0)  # Reset file pointer to the beginning
    return imghdr.what(None, image_data) is not None


@app.route('/')
def home():
    return 'Hello, this is your Flask app!'


@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    # Load known face encodings and names
    user_id = request.form['id']
    account_document = db.collection('Users').document(user_id)
    account_name = account_document.get().get('name')
    account_image_url = account_document.get().get('face')

    # Download the image from the URL
    response = requests.get(account_image_url)
    account_image = face_recognition.load_image_file(BytesIO(response.content))

    account_encoding = face_recognition.face_encodings(account_image)[0]

    known_face_encodings = [account_encoding]
    known_face_names = [account_name]

    try:
        # Receive image data from the React Native app
        image_file = request.files['file']

        # Check if the received file is a valid image
        if not is_valid_image(image_file):
            return jsonify({"error": "Invalid image file"}), 400

        unknown_encoding = face_recognition.face_encodings(face_recognition.load_image_file(image_file))[0]
    except IndexError:
        return jsonify({"error": "No face found. Please try again"}), 400

    # Assuming 'face' is the field storing the image URL
    label = test(image_file, model_dir="./Silent-Face-Anti-Spoofing/resources/anti_spoof_models", device_id=0)

    if label == 1:
        # See how far apart the test image is from the known faces
        face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)

        # Find the index of the closest known face
        min_distance_index = face_distances.argmin()

        if face_distances[min_distance_index] < tolerance:
            identified_name = known_face_names[min_distance_index]
            result = {"identified_name": identified_name}
        else:
            result = {"error": "Unrecognized face. Please try again"}
    else:
        result = {"error": "Are you Spoofing? Please try again"}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
