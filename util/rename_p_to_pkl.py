import os

PARENT_ROOT_DIR = 'playground/model_dir/'
assert os.getcwd().split('/')[-1] == 'cs221-pommerman', 'Please start the program from the cs221-pommerman/ directory.'


def rename_and_print(full_dir, file_prefix):
  old_name = file_prefix + '.p'
  old_path = os.path.join(full_dir, old_name)
  if os.path.exists(old_path):
    new_name = file_prefix + '.pkl'
    new_path = os.path.join(full_dir, new_name)
    os.rename(old_path, new_path)
    print('Moving {} -> {}'.format(old_path, new_path))


for project_name in os.listdir(PARENT_ROOT_DIR):
  full_dir = os.path.join(PARENT_ROOT_DIR, project_name)
  rename_and_print(full_dir, 'episode_recorder')
  rename_and_print(full_dir, 'reward_counter')
