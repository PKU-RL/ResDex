import os
import yaml
import subprocess

NUM_CLUSTER = 5

# create the log file in the clusters/str(NUM_CLUSTER) folder
os.makedirs('clusters/' + str(NUM_CLUSTER), exist_ok=True)
# if the log file already exists, delete it
if os.path.exists('clusters/' + str(NUM_CLUSTER) + '/log.txt'):
    os.system('rm clusters/' + str(NUM_CLUSTER) + '/log.txt')
os.system('touch clusters/' + str(NUM_CLUSTER) + '/log.txt')

# redirect print to append to the log file
log_file = open('clusters/' + str(NUM_CLUSTER) + '/log.txt', 'a')
print('Start training the model with ', NUM_CLUSTER, ' clusters!', file=log_file)
print('----------------------------------', file=log_file)
# load the clusters in txt file
clusters = []
for i in range(NUM_CLUSTER):
    cluster = []
    with open('clusters/'+str(NUM_CLUSTER) +'/cluster_'+str(i)+'.txt', 'r') as f:
        for line in f:
            object_code, scale = line.split()
            scale = float(scale)
            cluster.append({object_code: [scale]})
    clusters.append(cluster)


def change_training_object(obj):
    original_cfg = yaml.load(open('cfg/shadow_hand_blind_grasp.yaml', 'r'), Loader=yaml.FullLoader)
    original_cfg['env']['object_code_dict'].clear()
    original_cfg['env']['object_code_dict'] = obj 
    yaml.dump(original_cfg, open('cfg/shadow_hand_blind_grasp.yaml', 'w'))

def test_the_model(model_path):
    # find the line of ''--model_dir=' in the script
    # change the things after the '=' to the model_path + ' \' and a newline
    with open('script/mytest.sh', 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if '--model_dir=' in lines[i]:
            lines[i] = '--model_dir=' + model_path + ' \\\n'
            break
    with open('script/mytest.sh', 'w') as f:
        f.writelines(lines)
    # run the test script and get the last line of the output
    command = 'bash script/mytest.sh'
    output = subprocess.run(command, shell=True, capture_output=True)
    output = output.stdout.decode('utf-8').split('\n')
    # split the test result by ':' and get the last element
    result = output[-2].split(':')[-1]
    # get the success rate in the last element
    success_rate = float(result)
    return success_rate


for i in range(NUM_CLUSTER):
    for obj in clusters[i]:
        change_training_object(obj)
        success_rate = 0
        obj_name = list(obj.keys())[0]+'_'+str(list(obj.values())[0][0])
         # save the logs/test_seed0/model_5000.pt to model/base_model/10-means/obj.pt
        model_path = 'logs/test_seed0/model_5000.pt'
        save_path = 'model/base_model/'+str(NUM_CLUSTER)+'-means/' + obj_name + '.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print('Training object: ', obj, file=log_file)
        # check if the model already exists
        if os.path.exists(save_path):
            print('Model already exists: ', save_path, file=log_file)
            success_rate = test_the_model(save_path)
            print('The success rate is: ', success_rate, file=log_file)
            print('----------------------------------', file=log_file)
        
        else: 
            command = 'bash script/run_train_ppo_blind.sh '
            subprocess.run(command, shell=True)

        
            os.rename(model_path, save_path)
            print('Model saved to: ', save_path, file=log_file)
            print('----------------------------------', file=log_file)

            # test the model
            success_rate = test_the_model(save_path)

        if success_rate < 0.8:
            print(obj_name, success_rate, file=log_file)                    
            print('The model is not good enough, please retrain the model!', file=log_file)
            print('----------------------------------', file=log_file)
            continue
        else:
            print(obj_name, success_rate, file=log_file)
            print('The model is good enough, the success rate is: ', success_rate, file=log_file)
            print('----------------------------------', file=log_file)
            break


print('All models are trained and saved!', file=log_file)

