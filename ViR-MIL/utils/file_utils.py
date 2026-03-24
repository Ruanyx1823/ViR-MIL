import pickle
import h5py
import os

def save_pkl(filename, save_object):
    """保存pickle文件"""
    with open(filename, 'wb') as f:
        pickle.dump(save_object, f)

def load_pkl(filename):
    """加载pickle文件"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    """保存HDF5文件"""
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path