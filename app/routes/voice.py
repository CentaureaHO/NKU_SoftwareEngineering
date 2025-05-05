from flask import Blueprint, request, jsonify
from services.asr_service import process_audio

voice_bp = Blueprint('voice', __name__)

@voice_bp.route('/upload', methods=['POST'])
def upload_voice():
    audio = request.files.get('file')
    if not audio:
        return jsonify({'error': 'No file uploaded'}), 400
    result = process_audio(audio)
    return jsonify({'text': result})
