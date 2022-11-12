import os, pwd

uname = pwd.getpwuid(os.getuid())[0]

if uname == 'spolisetty':
    ROOT_DIR = "/home/spolisetty/OCC-GNN/cslicer/"
    SYSTEM = "jupiter"
if uname == 'q91':
    ROOT_DIR = "/home/q91/OCC-GNN/cslicer/"
    OUT_DIR = '/home/q91/OCC-GNN/experiments/exp6/'
    SYSTEM = "ornl"
if uname == 'spolisetty_umass_edu':
    ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN/cslicer"
    SRC_DIR = "/home/spolisetty_umass_edu/OCC-GNN/python/main.py"
    SYSTEM = 'unity'
    OUT_DIR = '/home/spolisetty_umass_edu/OCC-GNN/experiments/exp6/'

def get_git_info():
    repo = git.Repo(search_parent_directories = True)
    sha = repo.head.object.hexsha
    dirty = repo.is_dirty()
    return sha,dirty

def check_path():
    path_set = False
    for p in sys.path:
        if ROOT_DIR in p:
            path_set = True
    if (not path_set):
        # print(sys.path)
        sys.path.append(ROOT_DIR)
        # print("Setting Path")
        # print(sys.path)

check_path()
