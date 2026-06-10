import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer as mjv
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.intermimic import InterMimicRod
from utils.sim import get_obs_from_mujoco, pd_control


VIDEO_PLAYBACK_SPEED = 2.0


TASKS = {
    "box_plane": {
        "policy": "controller/policy/haic/box_plane/policy-hbrlhotq-final.onnx",
        "motion": "controller/motion/mirobotA/box_1",
        "xml": "assets/robots/g1/g1_29dof_rev_1_0_box.xml",
    },
    "box_slopestair": {
        "policy": "controller/policy/haic/box_terrians/policy-terrians.onnx",
        "motion": "controller/motion/mirobotA/box_slopestair",
        "xml": "assets/robots/g1/g1_29dof_rev_1_0_box_slopestair.xml",
    },
    "box_stairslope": {
        "policy": "controller/policy/haic/box_terrians/policy-terrians.onnx",
        "motion": "controller/motion/mirobotA/box_slopestair",
        "xml": "assets/robots/g1/g1_29dof_rev_1_0_box_stairslope.xml",
    },

    "pull_cart": {
        "policy": "controller/policy/haic/pull_cart/policy-ufisjrby-final.onnx",
        "motion": "controller/motion/mirobotA/pull_cart",
        "xml": "assets/robots/g1/g1_29dof_rev_1_0_pull_cart.xml",
    },
    "pull_cart_box": {
        "policy": "controller/policy/haic/pull_cart_box/policy-o338vk90-final.onnx",
        "motion": "controller/motion/mirobotA/pull_cart_box",
        "xml": "assets/robots/g1/g1_29dof_rev_1_0_pull_cart_box.xml",
    },    
    "push_cart": {
        "policy": "controller/policy/haic/push_cart/policy-pm5ymztg-final.onnx",
        "motion": "controller/motion/mirobotA/push_cart",
        "xml": "assets/robots/g1/g1_29dof_rev_1_0_push_cart.xml",
    },
    "push_wheelchair": {
        "policy": "controller/policy/haic/push_wheelchair/policy-jv9mbio2-final.onnx",
        "motion": "controller/motion/mirobotA/push_wheelchair",
        "xml": "assets/robots/g1/g1_29dof_rev_1_0_push_wheelchair.xml",
    },
    "skateboard": {
        "policy": "controller/policy/haic/skateboard/policy-9kykb93o-6000.onnx",
        "motion": "controller/motion/mirobotA/board6",
        "xml": "assets/robots/g1/g1_29dof_rev_1_0_skateboard.xml",
    },
}


RECOVERY_STAND_DEFAULT_POS = np.array([
    -0.2, 0.15, 0.0, 0.4, -0.2, 0.0,
    -0.2, -0.15, 0.0, 0.4, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0,
], dtype=np.float32)


class OpenCVVideoWriter:
    def __init__(self, path, fps, width, height):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Recording MP4 requires imageio[ffmpeg], imageio[pyav], or opencv-python. "
                "Install one of them, for example: pip install imageio[ffmpeg]"
            ) from exc

        self.cv2 = cv2
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"OpenCV could not open MP4 writer for {path}")

    def append_data(self, frame):
        self.writer.write(self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release()


def create_video_writer(path, fps, width, height, codec="h264"):
    if codec == "h264":
        try:
            import imageio
            import imageio_ffmpeg  # noqa: F401

            return imageio.get_writer(
                path,
                fps=fps,
                codec="libx264",
                macro_block_size=16,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not open an H.264 MP4 writer. Install the FFmpeg backend with:\n"
                "  pip install imageio-ffmpeg\n"
                "or run with --video_codec mp4v for the less-compatible OpenCV fallback."
            ) from exc

    try:
        import imageio

        return imageio.get_writer(path, fps=fps)
    except Exception as imageio_error:
        print(f"imageio could not open MP4 writer ({imageio_error}); falling back to OpenCV.")
        return OpenCVVideoWriter(path, fps, width, height)


def resolve_project_path(path):
    return (PROJECT_ROOT / path).resolve()


def validate_tasks(task_names=None):
    missing = []
    selected = task_names if task_names is not None else sorted(TASKS)
    for task_name in selected:
        task = TASKS[task_name]
        policy_path = resolve_project_path(task["policy"])
        motion_path = resolve_project_path(task["motion"])
        xml_path = resolve_project_path(task["xml"])

        if not policy_path.is_file():
            missing.append(f"{task_name}: missing policy {policy_path}")
        if not motion_path.is_dir():
            missing.append(f"{task_name}: missing motion folder {motion_path}")
        if not (motion_path / "meta.json").is_file():
            missing.append(f"{task_name}: missing motion meta.json")
        if not (motion_path / "motion.npz").is_file():
            missing.append(f"{task_name}: missing motion.npz")
        if not xml_path.is_file():
            missing.append(f"{task_name}: missing XML {xml_path}")

    if missing:
        raise FileNotFoundError("Task asset check failed:\n" + "\n".join(missing))


class HaicDeployment:
    def __init__(self, args):
        task = TASKS[args.task]
        self.task_name = args.task
        self.policy_path = resolve_project_path(task["policy"])
        self.motion_path = resolve_project_path(task["motion"])
        self.mjcf_path = resolve_project_path(task["xml"])
        self.sim_dt = args.sim_dt
        self.control_dt = args.control_dt
        self.viewer_dt = args.viewer_dt
        self.video_width = args.video_width
        self.video_height = args.video_height
        self.video_codec = args.video_codec
        self.num_actuators = args.num_actuators
        self.real_time = not args.no_real_time
        self.sim_time = 0.0
        self.next_viewer_time = 0.0
        self.policy_output = None
        self.sim_steps_per_control = args.sim_steps_per_control

        self.controller = InterMimicRod(self.policy_path, self.motion_path)
        self.control_steps = self.controller.refmotion.motion_steps
        self.sim_duration = self.control_steps * self.control_dt

        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.model.opt.timestep = self.sim_dt
        if args.record_video:
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.video_width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.video_height)
        self.data = mujoco.MjData(self.model)
        self.viewer = mjv.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 4.0
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 145

        self.renderer = (
            mujoco.Renderer(self.model, height=self.video_height, width=self.video_width)
            if args.record_video
            else None
        )
        self.record_video = args.record_video

        print(f"Task: {self.task_name}")
        print(f"Policy: {self.policy_path}")
        print(f"Motion: {self.motion_path}")
        print(f"MuJoCo XML: {self.mjcf_path}")
        print(f"Physics dt: {self.sim_dt} s ({1.0 / self.sim_dt:.1f} Hz)")
        print(f"Control dt: {self.control_dt} s ({1.0 / self.control_dt:.1f} Hz)")
        print(f"Sim steps per control step: {self.sim_steps_per_control}")
        print(f"Duration: {self.sim_duration:.3f} s ({self.control_steps} control steps)")
        if self.record_video:
            print(f"Video size: {self.video_width}x{self.video_height}")
            print(f"Video codec: {self.video_codec}")
            print(f"Video playback speed: {VIDEO_PLAYBACK_SPEED:g}x")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, 0.80]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:7 + self.num_actuators] = RECOVERY_STAND_DEFAULT_POS
        self.data.qvel[:] = 0.0
        self.controller.reset_history()
        mujoco.mj_forward(self.model, self.data)
        self.sim_time = 0.0
        self.next_viewer_time = 0.0
        self.policy_output = None

    def update_policy(self):
        state = get_obs_from_mujoco(self.data, self.num_actuators)
        self.policy_output = self.controller.step(state)

    def apply_pd_control(self):
        if self.policy_output is None:
            self.update_policy()

        qj = self.data.qpos[7:7 + self.num_actuators]
        dqj = self.data.qvel[6:6 + self.num_actuators]
        torque = pd_control(
            self.policy_output.target_dof_pos,
            qj,
            self.policy_output.dof_kps,
            np.zeros_like(self.policy_output.target_dof_pos),
            dqj,
            self.policy_output.dof_kds,
        )
        torque = np.clip(torque + self.policy_output.tau_ff, -self.controller.torque_limits, self.controller.torque_limits)
        self.data.ctrl[:self.num_actuators] = torque

    def step_simulator(self):
        self.apply_pd_control()
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.sim_dt

    def sync_viewer(self):
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id >= 0:
            self.viewer.cam.lookat[:] = self.data.xpos[pelvis_id]
        self.viewer.sync()

    def run(self):
        self.reset()
        mp4_writer = None
        try:
            if self.record_video:
                video_name = f"{self.task_name}_sim2sim.mp4"
                video_fps = max(1, int(round(VIDEO_PLAYBACK_SPEED / self.viewer_dt)))
                print(f"Saving video to {video_name}")
                mp4_writer = create_video_writer(
                    video_name,
                    video_fps,
                    self.video_width,
                    self.video_height,
                    self.video_codec,
                )

            for _ in tqdm(range(self.control_steps), desc="Simulating"):
                if not self.viewer.is_running():
                    break
                self.update_policy()

                for _ in range(self.sim_steps_per_control):
                    if not self.viewer.is_running():
                        break
                    start = time.perf_counter()
                    self.step_simulator()
                    should_render = self.sim_time + 1e-9 >= self.next_viewer_time
                    if should_render:
                        self.sync_viewer()
                        self.next_viewer_time += self.viewer_dt

                    if mp4_writer is not None and should_render:
                        self.renderer.update_scene(self.data, camera=self.viewer.cam)
                        mp4_writer.append_data(self.renderer.render())

                    elapsed = time.perf_counter() - start
                    if self.real_time and elapsed < self.sim_dt:
                        time.sleep(self.sim_dt - elapsed)
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved")
            self.viewer.close()
            print("Simulation finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="skateboard")
    parser.add_argument("--sim_dt", type=float, default=0.002)
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--viewer_dt", type=float, default=0.01)
    parser.add_argument("--num_actuators", type=int, default=29)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--video_codec", choices=["h264", "mp4v"], default="h264")
    parser.add_argument("--no_real_time", action="store_true")
    parser.add_argument("--check_assets", action="store_true", help="Check task policy/motion/XML triplets and exit")
    args = parser.parse_args()

    if args.sim_dt <= 0.0 or args.control_dt <= 0.0 or args.viewer_dt <= 0.0:
        parser.error("--sim_dt, --control_dt, and --viewer_dt must be positive")
    if args.video_width <= 0 or args.video_height <= 0:
        parser.error("--video_width and --video_height must be positive")
    if args.control_dt < args.sim_dt:
        parser.error("--control_dt must be greater than or equal to --sim_dt")
    sim_steps_per_control = round(args.control_dt / args.sim_dt)
    if not np.isclose(sim_steps_per_control * args.sim_dt, args.control_dt):
        parser.error("--control_dt must be an integer multiple of --sim_dt")
    args.sim_steps_per_control = sim_steps_per_control

    validate_tasks()
    if args.check_assets:
        print("Available task triplets:")
        for task_name, task in TASKS.items():
            print(f"  {task_name}: {task['policy']} + {task['motion']} + {task['xml']}")
        return

    HaicDeployment(args).run()


if __name__ == "__main__":
    main()
