import os
import pickle

def pickle3to2(full_dir, file_prefix):
  try:
    file_path = os.path.join(full_dir, file_prefix + '.pkl')
    file2_path = os.path.join(full_dir, file_prefix + '_2.pkl')
    with open(file_path, 'rb') as fin:
      data = pickle.load(fin)
      pickle.dump(data, open(file2_path, 'wb'), protocol=2)
  except:
    pass

ROOT_DIR = 'model_summary'
for project_name in os.listdir(ROOT_DIR):
  full_dir = os.path.join(ROOT_DIR, project_name)
  pickle3to2(full_dir, 'episode_recorder')
  pickle3to2(full_dir, 'reward_counter')
