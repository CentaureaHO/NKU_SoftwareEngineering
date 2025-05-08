from services.nlu_service import analyze_text

def fuse_modalities(text, vision_result):
    intent = analyze_text(text)
    if intent == "open_something" and "行人" not in vision_result:
        return "执行：打开天窗"
    return "保持当前状态"
