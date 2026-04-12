import unittest

from vla_demo.mission_loader import default_mission_path, load_mission_file
from vla_demo.state_machine import MissionStateMachine


class MissionStateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mission = load_mission_file(default_mission_path())
        self.machine = MissionStateMachine(robot_status_timeout_sec=5.0)
        self.now = 10.0
        self.machine.start_mission(self.mission, self.now)
        self.machine.handle_robot_status({"executor_state": "IDLE"}, self.now)

    def test_step_advances_after_required_votes(self) -> None:
        step_id = self.mission.steps[0].step_id
        self.machine.handle_inference(
            {
                "step_id": step_id,
                "step_match": True,
                "step_completed": True,
                "confidence": 0.95,
                "observed_landmarks": ["corridor"],
                "reason": "ok",
            },
            self.now + 2.1,
        )
        self.machine.handle_inference(
            {
                "step_id": step_id,
                "step_match": True,
                "step_completed": True,
                "confidence": 0.95,
                "observed_landmarks": ["corridor"],
                "reason": "ok",
            },
            self.now + 2.4,
        )
        self.machine.handle_inference(
            {
                "step_id": step_id,
                "step_match": True,
                "step_completed": True,
                "confidence": 0.95,
                "observed_landmarks": ["corridor"],
                "reason": "ok",
            },
            self.now + 2.8,
        )
        self.assertEqual(self.machine.current_step().step_id, 2)

    def test_timeout_pauses_and_requests_stop(self) -> None:
        step = self.machine.current_step()
        self.machine.tick(self.now + step.timeout_sec + 0.5)
        self.assertEqual(self.machine.mission_state, "PAUSED")
        self.assertEqual(self.machine.active_control_primitive(), "stop_and_hold")


if __name__ == "__main__":
    unittest.main()
