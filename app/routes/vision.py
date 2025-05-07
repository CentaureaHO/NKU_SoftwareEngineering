from flask import Blueprint, request, jsonify
from services.vision_service import process_image

vision_bp = Blueprint('vision', __name__)

@vision_bp.route('/upload', methods=['POST'])
def upload_image():
    image = request.files.get('file')
    if not image:
        return jsonify({'error': 'No image uploaded'}), 400
    result = process_image(image)
    return jsonify({'result': result})
