'''

def get_data_dir():
    import os
    username = os.environ['USER']
    if username == 'spolisetty_umass_edu':
        DATA_DIR = "/work/spolisetty_umass_edu/data"
        PATH_DIR = "/home/spolisetty_umass_edu/OCC-GNN"
    if username == "spolisetty":
        DATA_DIR = "/data/sandeep"
        PATH_DIR = "/home/spolisetty/OCC-GNN"
    if username == "q91":
        DATA_DIR = "/mnt/bigdata/sandeep"
        PATH_DIR = "/home/q91/OCC-GNN"
    return DATA_DIR,PATH_DIR

DATA_DIR, PATH_DIR = get_data_dir()
'''
