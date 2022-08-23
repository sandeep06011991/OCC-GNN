import numpy as np

ROOT_DIR = "/work/spolisetty_umass_edu/pagraph"

def print_char(filename):
    data_dir = "{}/{}".format(ROOT_DIR,filename)
    p_v = np.load("{}/{}".format(data_dir,"p_v.npy"),allow_pickle = True)
    total_vertices = np.load("{}/{}".format(data_dir, "labels.npy"), allow_pickle = True).shape[0]
    union_p_v = np.zeros(total_vertices)
    p_size = []
    for partition in p_v:
        union_p_v[partition] = 1
        p_size.append(partition.shape[0])
    used_vertices = np.sum(union_p_v)    
    frac_used = used_vertices/total_vertices    
    avg_overlap = sum(p_size)/(4 * used_vertices)
    print("Graph | total_vertices | frac_used | avg_overlap")
    print("{} | {} | {} | {}".format(filename, total_vertices, frac_used, avg_overlap))
    


'''
    Graph | total_vertices | frac_used | avg_overlap
    ogbn-arxiv | 169343 | 0.9791842591663075 | 0.7614281923554741
    ogbn-products | 2449029 | 0.9444020466887081 | 0.8818396683252135
    ogbn-papers100M | 111059956 | 0.11164376834437068 | 0.5792518915809727

'''
if __name__ == "__main__":
    filename = "ogbn-papers100M"
    print_char(filename)
