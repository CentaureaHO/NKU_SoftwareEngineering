from flask import Blueprint, request, jsonify

control_bp = Blueprint('control', __name__)

@control_bp.route('/execute', methods=['POST'])
def execute():
    intent = request.json.get('intent')
    return jsonify({'status': f'已执行指令：{intent}'})
