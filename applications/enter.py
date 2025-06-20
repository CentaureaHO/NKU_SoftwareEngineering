import time
from logger import logger
from viewer.viewer import update_note, update_light
from utils.tools import speecher_player
from system_init import get_component

class Enter:
    def __init__(self) -> None:
        self.metrics = {
            "gaze_pass_time": [],
            "modal_confirm_time": [],
            "decision_records": [],
            "total_rounds": 0,
            "success_rounds": 0,
            "modal_success": {
                "gaze": 0,
                "voice": 0,
                "voice_driver": 0,
                "gesture": 0,
                "headpose": 0,
            }
        }
        self.timing_info = {
            "系统初始化提示展示": 0,
            "视线方向检测完成": 0,
            "语音播报与提示完成": 0,
            "语音指令识别与处理": 0,
            "手势动作识别与响应": 0,
            "头部姿态识别与响应": 0,
            "功能确认融合判断与反馈": 0,
            "功能执行调用": 0,
        }

    def enter(self, controller) -> None:
        logger.Log("典型场景:启动系统时自动开启应用功能")
        application = get_component('application')
        setting = get_component('setting')

        # 阶段1：初始化提示
        t0 = time.time()
        update_note("系统初始化完毕，请驾驶员目视前方")
        update_light("red", True)
        self.timing_info["系统初始化提示展示"] = int((time.time() - t0) * 1000)

        # 阶段2：视线方向检测
        t0 = time.time()
        tag = False
        start_time = None
        while True:
            state = controller.gazer.get_key_info()
            if state != "中间":
                tag = False
                start_time = None
                continue
            if not tag:
                tag = True
                start_time = time.time()
            elif time.time() - start_time > 3:
                self.metrics["modal_success"]["gaze"] += 1
                break
        self.timing_info["视线方向检测完成"] = int((time.time() - t0) * 1000)

        # 阶段3：播报就位提示
        t0 = time.time()
        update_note("驾驶员已就位")
        update_light("green", False)
        speecher_player.speech_synthesize_sync("驾驶员已就位")
        self.timing_info["语音播报与提示完成"] = int((time.time() - t0) * 1000)

        func_list = [application.type.monitor_jump, application.type.navigation, application.type.music_play]
        str_full = ["监测车辆状态", "默认导航:从南开大学津南校区到八里台校区", "播放默认音乐:南开校歌"]
        str_short = ["监测", "导航", "播放音乐"]

        for i in range(len(func_list)):
            # 显示与播报语音询问
            text = f"是否为您{str_full[i]}\n同意请语音输入\"同意{str_short[i]}\"、竖起大拇指或点头\n拒绝请语音输入\"拒绝{str_short[i]}\"、摇手或摇头"
            update_note(text)
            update_light("green", True)
            speecher_player.speech_synthesize_sync(text)

            t_confirm_start = time.time()
            tag = None
            voice_time, gesture_time, headpose_time = 0, 0, 0

            while tag is None:
                # 语音输入
                t_voice = time.time()
                state = controller.speecher.get_key_info()
                driver_state = controller.speecher.update()
                is_driver = driver_state and driver_state.recognition.get("speaker_name") == setting.get_driver()
                if state and is_driver:
                    self.metrics["modal_success"]["voice"] += 1
                    self.metrics["modal_success"]["voice_driver"] += 1
                    if state == f"同意{str_short[i]}":
                        voice_time = time.time() - t_voice
                        tag = True
                        break
                    elif state == f"拒绝{str_short[i]}":
                        voice_time = time.time() - t_voice
                        tag = False
                        break

                # 手势输入
                t_gesture = time.time()
                state = controller.static_gesture_tracker.get_key_info()
                if state == "竖起大拇指":
                    gesture_time = time.time() - t_gesture
                    self.metrics["modal_success"]["gesture"] += 1
                    tag = True
                    break
                elif state == "摇手":
                    gesture_time = time.time() - t_gesture
                    self.metrics["modal_success"]["gesture"] += 1
                    tag = False
                    break

                # 头部姿态
                t_head = time.time()
                state = controller.headposer.get_key_info()
                if state == "点头":
                    headpose_time = time.time() - t_head
                    self.metrics["modal_success"]["headpose"] += 1
                    tag = True
                    break
                elif state == "摇头":
                    headpose_time = time.time() - t_head
                    self.metrics["modal_success"]["headpose"] += 1
                    tag = False
                    break

            # 记录耗时
            self.timing_info["语音指令识别与处理"] += int(voice_time * 1000)
            self.timing_info["手势动作识别与响应"] += int(gesture_time * 1000)
            self.timing_info["头部姿态识别与响应"] += int(headpose_time * 1000)
            self.timing_info["功能确认融合判断与反馈"] += int((time.time() - t_confirm_start) * 1000)

            # 功能执行
            t0 = time.time()
            update_light("green", False)
            if tag:
                update_note(f"正在为您{str_short[i]}")
                application.schedule(func_list[i], [])
                self.metrics["success_rounds"] += 1
                self.metrics["decision_records"].append("同意")
            else:
                update_note(f"您拒绝{str_short[i]}")
                self.metrics["decision_records"].append("拒绝")
            self.timing_info["功能执行调用"] += int((time.time() - t0) * 1000)
            self.metrics["total_rounds"] += 1

        self.report_metrics()

    def report_metrics(self):
        logger.Log("\n========= 多模态交互测试报告 =========")
        logger.Log(f"总轮数: {self.metrics['total_rounds']}")
        logger.Log(f"成功调度功能次数: {self.metrics['success_rounds']}")

        logger.Log("\n识别成功统计：")
        for k, v in self.metrics["modal_success"].items():
            logger.Log(f"{k:12s}: {v}")

        logger.Log("\n用户决策记录:")
        logger.Log(str(self.metrics["decision_records"]))

        logger.Log("\n关键阶段平均响应时间（ms）：")
        for k, v in self.timing_info.items():
            avg = v // self.metrics["total_rounds"] if "确认" in k or "调用" in k else v
            logger.Log(f"{k:24s}: {avg} ms")

        total = sum([
            v // self.metrics["total_rounds"]
            if "确认" in k or "调用" in k else v
            for k, v in self.timing_info.items()
        ])
        logger.Log(f"\n总流程平均耗时: {total} ms")
        logger.Log("=====================================\n")
