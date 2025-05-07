from flask import Blueprint, request, jsonify
from services.fusion_service import fuse_modalities

fusion_bp = Blueprint('fusion', __name__)

@fusion_bp.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    result = fuse_modalities(data.get('text'), data.get('vision'))
    return jsonify({'decision': result})
