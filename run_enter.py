import cProfile
import pstats
import io
import sys
import types
from applications.enter import Enter


# ---------- 模拟依赖组件 ----------
class FakeModality:
    def __init__(self, responses):
        self.responses = responses
        self.index = 0
    def get_key_info(self):
        if self.index < len(self.responses):
            val = self.responses[self.index]
            self.index += 1
            return val
        return self.responses[-1]
    def update(self):
        return type('Fake', (), {
            'recognition': {'speaker_name': 'driver'}
        })()


class FakeSetting:
    def get_driver(self):
        return "driver"


class FakeApplicationType:
    def __init__(self):
        self.monitor_jump = lambda: None
        self.navigation = lambda: None
        self.music_play = lambda: None


class FakeApplication:
    def __init__(self):
        self.type = FakeApplicationType()
    def schedule(self, func, args):
        func()


# ---------- 注入 system_init ----------
def inject_fake_components():
    fake_module = types.ModuleType("system_init")
    fake_module.get_component = lambda name: {
        "application": FakeApplication(),
        "setting": FakeSetting()
    }.get(name)
    sys.modules["system_init"] = fake_module


# ---------- 构造测试输入 ----------
def build_fake_controller():
    # 模拟 1 次偏移 + 20 次“中间”（1 秒 5 次，相当于稳定 4 秒）
    gaze_seq = ["左"] + ["中间"] * 20
    speech_seq = ["同意监测", "拒绝导航", "同意播放音乐"]

    return type('FakeController', (), {
        "gazer": FakeModality(gaze_seq),
        "speecher": FakeModality(speech_seq),
        "static_gesture_tracker": FakeModality([None, None, None]),
        "headposer": FakeModality([None, None, None])
    })()


# ---------- 运行性能分析 ----------
def profile_enter():
    inject_fake_components()
    controller = build_fake_controller()
    enter = Enter()

    pr = cProfile.Profile()
    pr.enable()

    enter.enter(controller)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(50)  # 打印前50个函数的耗时统计

    with open("profile_enter_report.txt", "w", encoding="utf-8") as f:
        f.write(s.getvalue())

    print("✅ 性能分析完成，结果已保存至 profile_enter_report.txt")


if __name__ == "__main__":
    profile_enter()
