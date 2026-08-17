import os
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

b = benchmark.get_benchmark_dict()
suite = b["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
print("ENV_OK", list(obs.keys()))
