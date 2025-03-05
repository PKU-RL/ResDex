# object_list should be a list not a dict, which is a mistake easy to be neglected
import yaml

object_list_dir="cfg/test_set_unseen_cat.yaml"
object_list = yaml.safe_load(open(object_list_dir, "r"))
object_list = object_list["object_code_list"]
object_code_dict = {}
for item in object_list:
    for key, value in item.items():
        if key in object_code_dict.keys():
            object_code_dict[key].extend(value)
        else:
            object_code_dict[key]=value

yaml.dump(object_code_dict, open('new_set.yaml', 'w'))