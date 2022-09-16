import torch
from data.bipartite import Bipartite
from data.part_sample import Sample
from data.part_sample import Gpu_Local_Sample
from dgl import DGLGraph

OBJECT_LIST = [Bipartite,Sample,Gpu_Local_Sample, torch.device]

IGNORED_LIST = [DGLGraph]

# todo: Add ordering to global object.
def get_attr_order_and_offset_size(object):
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
    objects.sort()
    assert(len(lists) < 2)
    # All integer offsets can be put into 1
    # Final size of the offset
    offset_size = len(tensors) +  len(lists) * 3 + (len(dictionary) * 4) + len(objects) + len(integers) + 1
    return (integers + tensors  + objects + dictionary + lists), offset_size

# global_order_dict = {}
# global_order_dict[type(Bipartite)] = get_attr_order_and_offset_size(Bipartite())
# global_order_dict[type(Gpu_Local_Sample)] = get_attr_order_and_offset_size(Gpu_Local_Sample())


def construct_from_tensor_on_gpu(tensor, device, object):
    print("ATtempting to reconstruct", type(object), tensor.shape)
    order, offset_size = get_attr_order_and_offset_size(object)
    # for i in global_order_dict[]
    print(tensor.device, device, tensor.dtype, "Dtype ")
    assert(tensor.device == device)
    # header Compute
    offsets = tensor[:offset_size]
    print("REConstructio !!!!!!!!!!!!!!!!!",order)

    data = tensor[offset_size:]
    offset_ptr = 0
    for  attr_name in order:
        attr_value = getattr(object, attr_name)
        print(attr_name, type(attr_value), offsets[offset_ptr], offsets[offset_ptr + 1])
        val_tensor = data[offsets[offset_ptr].item():offsets[offset_ptr + 1].item()]
        if type(attr_value) == (torch.Tensor):
            print(attr_name, val_tensor.shape, val_tensor.dtype)
            setattr(object, attr_name, val_tensor )
            offset_ptr = offset_ptr + 1
            continue
        if type(attr_value) == (int):
            assert(val_tensor.shape == (1,))
            setattr(object, attr_name, val_tensor.item())
            offset_ptr = offset_ptr + 1
            continue
        if type(attr_value) == torch.device:
            print("DEVICE #############",val_tensor.shape)
            setattr(object, attr_name, torch.device(val_tensor.item()))
            offset_ptr = offset_ptr + 1
            continue
        if type(attr_value) == list:
            print("list size", len(attr_value))
            for obj in (attr_value):
                val_tensor = data[offsets[offset_ptr].item():offsets[offset_ptr + 1].item()]
                constructed_object = construct_from_tensor_on_gpu(val_tensor,device, obj)
                # setattr(object, attr_name, constructed_object)
                offset_ptr = offset_ptr + 1
            continue
        #     assert(len(attr_value) == 3)
        #     for i in range(3):
        #         item = attr_value[i]
        #         tensor = serialize_to_tensor(item)
        #         data.append(tensor)
        #         offsets.append(offsets[-1] + tensor.shape[0])
        #     continue
        #         # How to handle this serializationn
        if type(attr_value) == type({}):
            d = {}
            for i in range(4):
                val_tensor = data[offsets[offset_ptr].item():offsets[offset_ptr + 1].item()]
                d[i] = val_tensor
                offset_ptr = offset_ptr + 1
            setattr(object, attr_name, d)
            continue
        if type(attr_value) in OBJECT_LIST:
            print("Recursive construction")
            constructed_object = construct_from_tensor_on_gpu(val_tensor,device, attr_value)
            setattr(object, attr_name, constructed_object)
            offset_ptr = offset_ptr + 1
            continue
        print("Coudlnt handle")
        print(attr_name, type(attr_value), list)
        assert(False)
    return object

def serialize_to_tensor(object):
    serialization_order, offset_size = get_attr_order_and_offset_size(object)
    data = []
    offsets = [0]
    print("TENSOR TO WRITE!!!!!!!!!!!!!!",serialization_order)
    for attr in serialization_order:
        attr_value = getattr(object, attr)
        if(type(attr_value) == torch.device):
            data.append(torch.tensor([attr_value.index]))
            offsets.append(offsets[-1] + 1)
            continue
        if(type(attr_value) == (int)):
            data.append(torch.tensor([attr_value]))
            offsets.append(offsets[-1] + 1)
            continue
        if type(attr_value) == (torch.Tensor):
            assert(len(attr_value.shape) == 1)
            data.append(attr_value)
            offsets.append(offsets[-1] + attr_value.shape[0])
            continue
        if type(attr_value) == list:
            assert(len(attr_value) == 3)
            for i in range(3):
                item = attr_value[i]
                assert(type(item) in OBJECT_LIST)
                tensor = serialize_to_tensor(item)
                assert(tensor.shape[0] >10)
                data.append(tensor)
                print("Adding to list", type(item), tensor.shape, i)
                offsets.append(offsets[-1] + tensor.shape[0])
            continue
                # How to handle this serializationn
        if type(attr_value) == type({}):
            assert(len(attr_value) == 4)
            for i in range(4):
                val = attr_value[i]
                assert(len(val.shape) == 1)
                data.append(attr_value[i])
                offsets.append(offsets[-1] + attr_value[i].shape[0])
            continue
        if type(attr_value) == DGLGraph:
            print("Heterograph neednt be serialized")
            continue
        if type(attr_value) in OBJECT_LIST:
            tensor_value = serialize_to_tensor(attr_value)
            data.append(tensor_value)
            offsets.append(offsets[-1] + tensor_value.shape[0])
            continue

        print(attr_value, attr, type(attr_value))
        # Unknown data data type
        raise Exception("Unknown object found")
    print(type(object), len(offsets), offset_size)
    assert(len(offsets) == offset_size)
    meta = torch.tensor(offsets)
    final_data = []
    final_data.append(meta)
    final_data.extend(data)
    # print(final_data)
    res = torch.cat(final_data, dim = 0).int()
    return res
