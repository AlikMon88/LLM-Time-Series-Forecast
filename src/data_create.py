
import h5py

def load_data(file_path, time_step_split, is_plot = False):
    with h5py.File(file_path, "r") as f:
        # Print the dataset keys
        print("Keys in HDF5 file:", list(f.keys()))
        time_data, traj_data = f[list(f.keys())[0]][:], f[list(f.keys())[-1]][:]
    
    data_prey, data_pred = traj_data[:, :, 0], traj_data[:, :, -1]

    data_prey, data_prey_true = data_prey[:, :int(time_step_split*data_prey.shape[-1])], data_prey[:, int(time_step_split*data_prey.shape[-1]) - 1:]
    data_pred, data_pred_true = data_pred[:, :int(time_step_split*data_pred.shape[-1])], data_pred[:, int(time_step_split*data_pred.shape[-1]) - 1:]
    time_data_past, time_data_true = time_data[:int(time_step_split*time_data.shape[0])], time_data[int(time_step_split*time_data.shape[0]) - 1:]

    if is_plot:
        return data_prey, data_prey_true, data_pred, data_pred_true, time_data_past, time_data_true
    
    else:
        return data_prey, data_prey_true, data_pred, data_pred_true
