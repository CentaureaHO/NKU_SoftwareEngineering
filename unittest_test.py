import unittest
from unittest.mock import patch, MagicMock
import time

from applications.enter import Enter  
import itertools

class TestEnterFunction(unittest.TestCase):
    @patch("applications.enter.update_note")
    @patch("applications.enter.update_light")
    @patch("applications.enter.speecher_player.speech_synthesize_sync")
    @patch("applications.enter.time.time", side_effect=[
        0, 1, 2, 4,  # gaze 连续为 "中间"，超过3秒
        5, 6, 7, 8, 9, 10, 11, 12  # 后续三轮同意
    ])
    @patch("system_init.get_component")
    def test_enter_driver_accepts_all(self, mock_get_component, mock_time, mock_speak, mock_light, mock_note):
        mock_app = MagicMock()
        mock_set = MagicMock()
        mock_set.get_driver.return_value = "test_user"
        mock_get_component.side_effect = lambda name: {
            "application": mock_app,
            "setting": mock_set
        }[name]

        controller = MagicMock()
        controller.gazer.get_key_info.side_effect = itertools.repeat("中间")
        controller.speecher.get_key_info.side_effect = ["同意监测", "同意导航", "同意播放音乐"]
        
        class Dummy:
            recognition = {"speaker_name": "test_user"}
        controller.speecher.update.side_effect = [Dummy()] * 3
        controller.static_gesture_tracker.get_key_info.return_value = None
        controller.headposer.get_key_info.return_value = None

        enter = Enter()
        enter.enter(controller)

        self.assertEqual(mock_app.schedule.call_count, 3)
        # mock_speak.assert_any_call("系统初始化完毕，请驾驶员目视前方")
        # mock_speak.assert_any_call("正在为您播放音乐")