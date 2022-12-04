
def normalize(epoch, sample_get, movement_data,\
            forward_time, backward_time):
    pred_total = float(sample_get) + float(movement_data)  + float(forward_time) + float(backward_time)
    f = float(epoch)/float(pred_total)
    sample_get = float(sample_get) * f
    movement_data_time = (float(movement_data)) * f
    forward_time = float(forward_time) * f
    backward_time = float(backward_time) * f

    return sample_get, movement_data_time, forward_time, backward_time 

