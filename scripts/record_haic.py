import os
import sys


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    if "--record_video" not in sys.argv:
        sys.argv.append("--record_video")
    if "--no_real_time" not in sys.argv:
        sys.argv.append("--no_real_time")

    from deploy_haic import main as deploy_main

    deploy_main()


if __name__ == "__main__":
    main()
