import cProfile
import pstats
import io
import sys
import types
from multimodal_controller import MultimodalController

# 动态注册 individuation 到 system_init.get_component 中
def inject_dummy_individuation():
    class DummyIndividuation:
        def gesture_individuation(self, info):
            print(f"[Dummy] 处理手势识别结果: {info}")
        def speech_individuation(self, info):
            print(f"[Dummy] 处理语音识别结果: {info}")

    # 模拟 system_init.py 中的 get_component 方法
    dummy_module = types.ModuleType("system_init")
    dummy_module.get_component = lambda name: DummyIndividuation() if name == "individuation" else None

    sys.modules["system_init"] = dummy_module
    print("[注入] 已注入 DummyIndividuation 到 system_init.get_component")


def run_controller():
    inject_dummy_individuation()
    controller = MultimodalController()
    controller.control()


def profile_run():
    pr = cProfile.Profile()
    pr.enable()

    try:
        run_controller()
    except KeyboardInterrupt:
        print("退出多模态控制主循环")

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(50)

    with open("profile_report.txt", "w") as f:
        f.write(s.getvalue())
    print("[性能测试] 分析报告已保存至 profile_report.txt")


if __name__ == "__main__":
    profile_run()
