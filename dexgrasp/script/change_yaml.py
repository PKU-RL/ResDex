import yaml

#destination_dir = "cfg/shadow_hand_pcl.yaml"
#destination_dir = "cfg/shadow_hand_residual_grasp.yaml"
#destination_dir = "cfg/shadow_hand_grasp.yaml"
destination_dir = "cfg/shadow_hand_blind_grasp.yaml"

source_dir = "cfg/new_train_set.yaml"
#source_dir = "cfg/new_seen_cat.yaml"
#source_dir = "cfg/new_unseen_cat.yaml"

destination = yaml.safe_load(open(destination_dir, "r"))
source = yaml.safe_load(open(source_dir, "r"))

destination["env"]["object_code_dict"] = source

yaml.dump(destination, open(destination_dir, "w"))
