import torch
from data.bipartite import Bipartite
from data.part_sample import Sample
from data.part_sample import Gpu_Local_Sample
from dgl import DGLGraph

OBJECT_LIST = [Bipartite,Sample,Gpu_Local_Sample, torch.device]
# I dont have to serialize the graph
IGNORED_LIST = [DGLGraph]

# todo: Add ordering to global object.
def get_attr_order_and_offset_size(object, num_partitions = 4):
    tensors = []
    integers = []
    dictionary = []
    objects = []
    lists = []
    for attr in dir(object):
        val = getattr(object, attr)
        if not callable(val) \
            and not attr.startswith("__"):
            if type(val) == torch.Tensor:
                tensors.append(attr)
                continue
            if type(val) == int:
                integers.append(attr)
                continue
            if type(val) == type({}):
                dictionary.append(attr)
                continue
            if type(val) in OBJECT_LIST:
                objects.append(attr)
                continue
            if type(val) in IGNORED_LIST:
                continue
            if type(val) == list:
                lists.append(attr)
                continue
            # Dont handle silently.
            print(attr,val,type(val))
            assert(False)
    tensors.sort()
    integers.sort()
    dictionary.sort()
    lists.sort()
    objects.sort()


    # All integer offsets can be put into 1
    # Final size of the offset
    offset_size = len(tensors) + len(lists) + (len(dictionary) * num_partitions) + len(objects) + len(integers) + 1
    return (integers + tensors  + objects + dictionary + lists ), offset_size

global_order_dict = {}
global_order_dict[Bipartite] = get_attr_order_and_offset_size(Bipartite())
global_order_dict[Gpu_Local_Sample] = get_attr_order_and_offset_size(Gpu_Local_Sample())
# print(global_order_dict.keys())


def construct_from_tensor_on_gpu(tensor, device, object, num_gpus = 4):
    # print("Warning. Dont have to reanalyze the object everytime. ")
    # order, offset_size = get_attr_order_and_offset_size(object)
    order, offset_size = global_order_dict[type(object)]
    # assert(offset_size == global_order_dict[type(object)][1])
    # assert(len(order) == len(global_order_dict[type(object)][0]))
    assert(tensor.device == device)
    
    # header Compute
    offsets = tensor[:offset_size].tolist()
    data = tensor[offset_size:]
    offset_ptr = 0
    for  attr_name in order:
        attr_value = getattr(object, attr_name)
        val_tensor = data[offsets[offset_ptr]:offsets[offset_ptr + 1]]
        if type(attr_value) == (torch.Tensor):
            setattr(object, attr_name, val_tensor )
            offset_ptr = offset_ptr + 1
            continue
        if type(attr_value) == (int):
            assert(val_tensor.shape == (1,))
            setattr(object, attr_name, val_tensor.item())
            offset_ptr = offset_ptr + 1
            continue
        if type(attr_value) == torch.device:
            setattr(object, attr_name, torch.device(val_tensor.item()))
            offset_ptr = offset_ptr + 1
            continue
        if type(attr_value) == list:
            object_type = type(getattr(object,attr_name)[0])
            list_data = data[offsets[offset_ptr] : offsets[offset_ptr + 1]]
            if (object_type) == int:
                ls = list_data.tolist()
                length = ls[0]
                final_value = ls[1:]
                # for i in list_data[1:]:
                #     final_value.append(i.item())
                assert(length == len(final_value))
            else:
                length = list_data[0]
                final_value = []
                tensor_list_data = list_data[length + 1 +1:]
                for i in range(length):
                    start = list_data[i+1]
                    end = list_data[i+2]
                    data_ = tensor_list_data[start:end]
                    final_value.append(construct_from_tensor_on_gpu(data_, device, object_type(), num_gpus = num_gpus))
            setattr(object, attr_name, final_value)
            offset_ptr = offset_ptr + 1
            continue
        if type(attr_value) == type({}):
            d = {}
            for i in range(num_gpus):
                val_tensor = data[offsets[offset_ptr] :offsets[offset_ptr + 1]]
                d[i] = val_tensor
                offset_ptr = offset_ptr + 1
            setattr(object, attr_name, d)
            continue
        if type(attr_value) in OBJECT_LIST:
            if(val_tensor.shape[0]):
                constructed_object = None
                setattr(object, attr_name, constructed_object)
            else:
                constructed_object = construct_from_tensor_on_gpu(val_tensor,device, attr_value, num_gpus = num_gpus)
                setattr(object, attr_name, constructed_object)
            offset_ptr = offset_ptr + 1
            continue
        assert(False)
    return object

def serialize_to_tensor(object, device = torch.device('cpu'), num_gpus = 4):
    serialization_order, offset_size =  global_order_dict[type(object)]
    data = []
    offsets = [0]
    for attr in serialization_order:
        attr_value = getattr(object, attr)
        # print("serialize",attr, offsets, attr_value)
        if attr_value == None:
            offsets.append(offsets[-1])
            continue
        if(type(attr_value) == torch.device):
            data.append(torch.tensor([attr_value.index], device = device))
            offsets.append(offsets[-1] + 1)
            continue
        if(type(attr_value) == (int)):
            data.append(torch.tensor([attr_value], device = device))
            offsets.append(offsets[-1] + 1)
            continue
        if type(attr_value) == (torch.Tensor):
            assert(len(attr_value.shape) == 1)
            data.append(attr_value)
            offsets.append(offsets[-1] + attr_value.shape[0])
            continue
        if type(attr_value) == list:
            local_tensors = []
            length = torch.tensor([len(attr_value)], device = device)
            object_type = type(attr_value[0])
            if object_type  == int:
                local_tensors.append(length)
                local_tensors.append(torch.tensor(attr_value, device = device))
                # print(local_tensors)
                local_tensors = torch.cat(local_tensors, dim = 0)
            else:
                objects_serialized = []
                local_offsets = [0]
                for i in range(len(attr_value)):
                    item = attr_value[i]
                    assert(type(item) in OBJECT_LIST)
                    tensor = serialize_to_tensor(item,device, num_gpus = num_gpus)
                    objects_serialized.append(tensor)
                    local_offsets.append(local_offsets[-1] + tensor.shape[0])
                local_tensors = torch.cat([length, torch.tensor(local_offsets, device = device)] + objects_serialized, dim = 0)
            data.append(local_tensors)
            offsets.append(offsets[-1] + local_tensors.shape[0])
            continue
        if type(attr_value) == type({}):
            assert(len(attr_value) == num_gpus)
            for i in range(num_gpus):
                val = attr_value[i]
                assert(len(val.shape) == 1)
                data.append(attr_value[i])
                offsets.append(offsets[-1] + attr_value[i].shape[0])
            continue
        if type(attr_value) == DGLGraph:
            print("Heterograph neednt be serialized")
            continue
        if type(attr_value) in OBJECT_LIST:
            tensor_value = serialize_to_tensor(attr_value, device, num_gpus= num_gpus)
            data.append(tensor_value)
            offsets.append(offsets[-1] + tensor_value.shape[0])
            continue
        print(attr_value, attr, type(attr_value))
        # Unknown data data type
        raise Exception("Unknown object found")
    assert(len(offsets) == offset_size)
    meta = torch.tensor(offsets, device = device)
    final_data = []
    final_data.append(meta)
    final_data.extend(data)
    res = torch.cat(final_data, dim = 0).to(torch.int32)
    assert(res.dtype == torch.int)
    return res
