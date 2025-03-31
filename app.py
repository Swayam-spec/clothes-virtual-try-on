from flask import Flask, request, jsonify, render_template
from PIL import Image
import requests
from io import BytesIO
import base64
import sys
import os  # Import the os module

app = Flask(__name__)

if sys.platform == "win32":
    import msvcrt
    # Use msvcrt for file locking or other operations
else:
    import fcntl
    # Use fcntl for file locking or other operations

# Create uploads directory if it doesn't exist
uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    if 'cloth' not in request.files or 'model' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    cloth_file = request.files['cloth']
    model_file = request.files['model']

    # Save the files to the uploads directory
    cloth_file.save(os.path.join(uploads_dir, cloth_file.filename))
    model_file.save(os.path.join(uploads_dir, model_file.filename))

    # Here you would typically run your model and get the output
    output_image = "base64_encoded_image_string_here"  # Replace with actual output

    return jsonify({'op': output_image})  # Return the output as JSON

if __name__ == '__main__':
    app.run(debug=True)
