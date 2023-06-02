import copy
import json
import jsonpickle
import shutil
import os

def copy_and_apply(src, deep=False, **kwargs):
    if deep:
        cpy = copy.deepcopy(src)
    else:
        cpy = copy.copy(src)
    for k, v in kwargs.items():
        setattr(cpy, k, v)
    return cpy

def to_string(obj):
    return jsonpickle.encode(obj)

def convert_json(obj):
    #Convert obj to a version which can be serialized with JSON.
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) 
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) 
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False

def load_last_model(fpath):
    if not os.path.exists(fpath):
        itr = 0
    else:
        # Checkpoint are saved as rl_model_XXXXXX_steps.zip
        checkpoints = [int(x[9:-10]) for x in os.listdir(fpath) if x.endswith('.zip')]
        if not len(checkpoints):
            itr = 0
            traj_dir = os.path.join(os.path.dirname(fpath), 'trajectory_logger')
            delete_trajectory__files(traj_dir, itr)
        else:
            itr = max(checkpoints)
        
    return itr

def delete_trajectory__files(fpath, itr):
    if os.path.exists(fpath):

        # trajectories are stored in trajectory_logger file as episode_XXX
        # equal is used because first episode is 0.
        itr_episodes = [int(x[8:]) for x in os.listdir(fpath) if int(x[8:]) >= itr]
        episodes = ['episode_' + str(number) for number in itr_episodes]

        for episode in episodes:
            shutil.rmtree(os.path.join(fpath, episode))
