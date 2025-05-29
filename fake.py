# run_security_test.py

from applications.enter import Enter
from system_init import register_component, get_component
import types
import threading
import time
import os

# ==== Step 1: 注册 fake setting（返回合法驾驶员） ====
class FakeSetting:
    def get_driver(self):
        return "driver"

register_component("setting", FakeSetting())

# ==== Step 2: 注册 fake application（拦截调度） ====
called_flag = {"called": False}

class FakeApp:
    def __init__(self):
        self.type = types.SimpleNamespace(
            monitor_jump=lambda: print("mock monitor_jump"),
            navigation=lambda: print("mock navigation"),
            music_play=lambda: print("mock music_play")
        )

    def schedule(self, func, args):
        called_flag["called"] = True
        print("非法调度被触发！系统可能存在权限漏洞")
        func()

register_component("application", FakeApp())

# ==== Step 3: 模拟非法语音用户 attacker ====
class FakeSpeecher:
    def __init__(self):
        self.call_count = 0

    def get_key_info(self):
        self.call_count += 1
        if self.call_count == 1:
            return "同意导航"  # 非法用户尝试触发任务
        elif self.call_count < 20:
            return None
        else:
            raise KeyboardInterrupt()  # 超过次数自动中断

    def update(self):
        return types.SimpleNamespace(recognition={"speaker_name": "attacker"})

# ==== Step 4: 模拟其他模态模块 ====
class DummyModule:
    def get_key_info(self):
        return None

# ==== Step 5: 构造 Controller ====
class FakeController:
    def __init__(self):
        self.speecher = FakeSpeecher()
        self.static_gesture_tracker = DummyModule()
        self.headposer = DummyModule()
        self.gazer = types.SimpleNamespace(get_key_info=lambda: "中间")

# ==== Step 6: 超时保护线程 ====
def timeout_guard(seconds=10):
    time.sleep(seconds)
    print(f"超时退出（超过 {seconds}s 未完成）")
    os._exit(0)

# ==== Step 7: 执行测试 ====
if __name__ == '__main__':
    threading.Thread(target=timeout_guard, daemon=True).start()

    try:
        Enter().enter(FakeController())
    except KeyboardInterrupt:
        print("自动中断：测试结束")

    # ==== Step 8: 判断安全性结果 ====
    print("\n测试结果：")
    if called_flag["called"]:
        print("安全性测试失败：非法用户触发调度！")
    else:
        print("安全性测试通过：非法用户未能触发调度。")
