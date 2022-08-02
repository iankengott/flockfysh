import os
import re

def get_exp_dir(exp_upper_dir):
	cur_num  = -1

	for folder in os.listdir(exp_upper_dir):
		if os.path.exists(os.path.join(exp_upper_dir, folder, 'weights', 'best.pt')):
			pot_num = (1 if folder == 'exp' else int(re.findall(r'\d+$', folder)[-1]))

			if pot_num > cur_num:
				cur_num = pot_num

	if cur_num == -1:
		raise Exception("No yolo directory found")

	return f'exp{cur_num}' if cur_num != 1 else 'exp'
