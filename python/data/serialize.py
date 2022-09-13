import torch
from data.bipartite import Bipartite
from data.part_sample import Sample
from data.part_sample import Gpu_Local_Sample
from dgl import DGLGraph

OBJECT_LIST = [Bipartite,Sample,Gpu_Local_Sample, DGLGraph]

def get_attr_order_and_offset_size(object):
    tensors = []
    integers = []
    dictionary = []
    objects = []
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
            # Dont handle silently.
            print(attr,val,type(val))
            assert(False)

    tensors.sort()
    integers.sort()
    dictionary.sort()
    objects.sort()
    # All integer offsets can be put into 1
    # Final size of the offset
    offset_size = len(tensors) + (len(dictionary) * 4) + len(objects) + integers + 1
    return (tensors + integers + objects + dictionary), offset_size

def construct_from_tensor_on_gpu(self, tensor, device, object):
    order, offset_size = get_attr_order_and_offset_size(object)
    assert(tensor.device == device)
    # header Compute
    offsets = tensor[:offset_size]
    data = tensor[offset_size:]
    offset_ptr = 0
    for  attr_name in order:
        attr_value = getattr(object, attr)
        val_tensor = data[offsets[offset_ptr]:offsets[offset_ptr + 1]]
        if type(attr_value) == type(torch.tensor):
            setAttr(object, attr_name, val_tensor )
        if type(attr_value) == type(int):
            assert(val_tensor.shape == (1,))
            setAttr(object, attr_name, val_tensor.item())
        if type(attr_value) in OBJECT_LIST:
            constructed_object = construct_from_tensor_on_gpu(val_tensor,device, attr_value)
            setAttr(object, attr_name, constructed_object)

            continue


def serialize_to_tensor(object):
    serialization_order, offset_size = get_attr_order_and_offset_size(object)
    data = []
    offsets = [0]
    for attr in serialization:
        attr_value = getattr(object, attr)
        if type(attr_value) == type(torch.tensor):
            assert(len(attr_value.shape) == 1)
            data.append(attr_value)
            offsets.append(offsets[-1] + data.shape[0])
            continue
        if type(attr_value) == type({}):
            assert(len(attr_value) == 4)
            for i in range(4):
                val = attr_value[i]
                assert(len(val.shape) == 1)
                data.append(attr_value)
                offsets.append(offsets[-1] + data.shape[0])
            continue
        if type(attr_value) in OBJECT_LIST:
            tensor = serialize_to_tensor(attr_value)
            data.append(tensor_value)
            offsets.append(offsets[-1] + data.shape[0])
            continue
        if type(attr_value) in dgl.heterograph:
            print("Heterograph neednt be serialized")
            continue

        # Unknown data data type
        raise Exception("Unknown object found")
    assert(len(offsets) == offset_size)
    meta = torch.tensor(offsets)
    final_data = []
    final_data.append(meta)
    final_data.extend(data)
    res = torch.cat(final_data, dim = 0)
    return res
