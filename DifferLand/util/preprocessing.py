import xarray as xr
import numpy as np
import collections
import os
import jax.numpy as jnp
import pandas as pd
from datetime import date, timedelta
from copy import deepcopy
import jax
from typing import Optional

def read_variable_to_vector(dir_name, nc_filename, var_name, not_nan_idx=None, shuffle_idx=None, time_idx=None):
    if var_name == "HEMISPHERE_BINARY":
        if time_idx is not None:
            arr = xr.open_dataset(os.path.join(dir_name, nc_filename))["LAT"].values[time_idx, :, :].flatten()
            arr[arr >= 0] = 1.0
            arr[arr < 0] = -1.0
        else:
            arr = xr.open_dataset(os.path.join(dir_name, nc_filename))["LAT"].values.flatten()
            arr[arr >= 0] = 1.0
            arr[arr < 0] = -1.0
    elif var_name=="LAT_SIGMOID":
        if time_idx is not None:
            arr = xr.open_dataset(os.path.join(dir_name, nc_filename))["LAT"].values[time_idx, :, :].flatten()
            arr = np.array(jax.nn.sigmoid(arr))
        else:
            arr = xr.open_dataset(os.path.join(dir_name, nc_filename))["LAT"].values.flatten()
            arr = np.array(jax.nn.sigmoid(arr))
    else:    
        if time_idx is not None:
            arr = xr.open_dataset(os.path.join(dir_name, nc_filename))[var_name].values[time_idx, :, :].flatten()
        else:
            arr = np.array(xr.open_dataset(os.path.join(dir_name, nc_filename))[var_name].values.flatten())
    
    if isinstance(not_nan_idx, (collections.abc.Sequence, np.ndarray)):
        arr = arr[not_nan_idx]
    if isinstance(shuffle_idx, (collections.abc.Sequence, np.ndarray)):
        arr = arr[shuffle_idx]
        
    return arr

def read_multiple_varible_to_array(dir_name, nc_filename, var_name_list):
    return np.stack([read_variable_to_vector(dir_name, nc_filename, var_name) for var_name in var_name_list])  

def read_single_variable_temporal_to_vector(dir_name, nc_filename, var_name, not_nan_idx, shuffle_idx):
    data_arr = xr.open_dataset(os.path.join(dir_name, nc_filename))[var_name].values
    data_arr=data_arr.reshape(data_arr.shape[0], data_arr.shape[1] * data_arr.shape[2])
    data_arr = data_arr[:, not_nan_idx][:, shuffle_idx]
    return data_arr

def build_temporal_mat_from_static(data_ar, n_t):
    data_mat = np.zeros((n_t, data_ar.shape[0]), dtype=np.float32)
    for i in range(n_t):
        data_mat[i, :] = data_ar
    return data_mat

def read_multiple_variable_temporal_to_vector(dir_name, nc_filename, var_name_list, not_nan_idx,
                                              shuffle_idx, co2_filename= "co2_mm_gl_01_23.csv", n_t=108):
    data_list = []
    
    for var_name in var_name_list:
        if var_name == "CO2":
            co2 = pd.read_csv(os.path.join(dir_name, co2_filename)).average
            co2_ar = np.zeros((n_t, len(shuffle_idx)), dtype=np.float32)
            for i in range(n_t):
                co2_ar[i, :]=co2[i]
            data_list.append(co2_ar)
        elif var_name == "LAT":
            data_ar = read_variable_to_vector(dir_name, nc_filename, "LAT", not_nan_idx, shuffle_idx)
            data_list.append(build_temporal_mat_from_static(data_ar, n_t))
        elif var_name == "DELTA_T":            
            deltat_ar = np.full((n_t, len(shuffle_idx)), 30, dtype=np.float32)
            data_list.append(deltat_ar)
        elif var_name == "MAT":
            data_ar = read_variable_to_vector(dir_name, nc_filename, "MAT", not_nan_idx, shuffle_idx) 
            data_list.append(build_temporal_mat_from_static(data_ar, n_t))
        elif var_name == "MAP":
            data_ar = read_variable_to_vector(dir_name, nc_filename, "MAP", not_nan_idx, shuffle_idx) / 365.22
            data_list.append(build_temporal_mat_from_static(data_ar, n_t))
        else:
            data_ar = read_single_variable_temporal_to_vector(dir_name, nc_filename,
                                                                     var_name, not_nan_idx, shuffle_idx)
            data_list.append(data_ar)
    return jnp.stack(data_list)


def nan_read_multiple_variable_temporal_to_vector(dir_name, nc_filename, var_name_list, not_nan_idx,
                                              shuffle_idx, n_t=108):
    data_list = []
    
    for var_name in var_name_list:
        if var_name == "som_const":
            data_vec = read_variable_to_vector(dir_name, nc_filename, "initial_som", not_nan_idx, shuffle_idx)
            data_ar = build_temporal_mat_from_static(data_vec, n_t)
            data_ar_valid = np.invert(np.isnan(data_ar)).astype(np.float32)
        elif var_name == "biomass_yan":
            data_ar = read_single_variable_temporal_to_vector(dir_name, nc_filename, "agb_yan", not_nan_idx, shuffle_idx)
            data_ar += read_single_variable_temporal_to_vector(dir_name, nc_filename, "bgb_yan", not_nan_idx, shuffle_idx)  
            data_ar_valid = np.invert(np.isnan(data_ar)).astype(np.float32)
            
        elif var_name == "biomass_ib":
            data_ar = read_single_variable_temporal_to_vector(dir_name, nc_filename, "IB_AGC", not_nan_idx, shuffle_idx)
            data_ar += read_single_variable_temporal_to_vector(dir_name, nc_filename, "IB_BGC", not_nan_idx, shuffle_idx)  
            data_ar_valid = np.invert(np.isnan(data_ar)).astype(np.float32)
        elif var_name == "gpp_fluxnet" or var_name == "reco_fluxnet" or var_name == "et_fluxnet":
            fluxnet_weight = read_single_variable_temporal_to_vector(dir_name, nc_filename,
                                                                     var_name+"_weight", not_nan_idx, shuffle_idx)
            data_ar = read_single_variable_temporal_to_vector(dir_name, nc_filename, var_name, not_nan_idx, shuffle_idx)
            data_ar_valid = np.invert(np.isnan(data_ar)).astype(np.float32)
            data_ar_valid[fluxnet_weight>0] = fluxnet_weight[fluxnet_weight>0]
        elif var_name == "gpp_fluxnet_10percent" or var_name == "reco_fluxnet_10percent" or var_name == "et_fluxnet_10percent":
            fluxnet_weight = read_single_variable_temporal_to_vector(dir_name, nc_filename,
                                                                     "_".join(var_name.split("_")[:-1])+"_weight_10percent",
                                                                     not_nan_idx, shuffle_idx)
            data_ar = read_single_variable_temporal_to_vector(dir_name, nc_filename, var_name, not_nan_idx, shuffle_idx)
            data_ar_valid = np.invert(np.isnan(data_ar)).astype(np.float32)
            data_ar_valid[fluxnet_weight>0] = fluxnet_weight[fluxnet_weight>0]
        elif var_name == "gpp_fluxnet_50percent" or var_name == "reco_fluxnet_50percent" or var_name == "et_fluxnet_50percent":
            fluxnet_weight = read_single_variable_temporal_to_vector(dir_name, nc_filename,
                                                                     "_".join(var_name.split("_")[:-1])+"_weight_50percent",
                                                                     not_nan_idx, shuffle_idx)
            data_ar = read_single_variable_temporal_to_vector(dir_name, nc_filename, var_name, not_nan_idx, shuffle_idx)
            data_ar_valid = np.invert(np.isnan(data_ar)).astype(np.float32)
            data_ar_valid[fluxnet_weight>0] = fluxnet_weight[fluxnet_weight>0]
        elif var_name == "none": # mask the output entirely when it is not used
            data_ar = np.full((n_t, len(shuffle_idx)), np.nan, dtype=np.float32)
            data_ar_valid = np.invert(np.isnan(data_ar)).astype(np.float32)
        else:
            data_ar = read_single_variable_temporal_to_vector(dir_name, nc_filename, var_name, not_nan_idx, shuffle_idx)
            data_ar_valid = np.invert(np.isnan(data_ar)).astype(np.float32)
        data_ar[np.isnan(data_ar)] = -9999
        data_list.append(data_ar)
        data_list.append(data_ar_valid)
        print(var_name)
        print(data_ar.shape)
        print(data_ar_valid.shape)
    return jnp.stack(data_list)

def generate_data_loader(data_matrix, idx_list, batch_size=320, zero_padding=True):
    DIVIDE_BATCH_LIST = []
    for i in np.unique(idx_list):
        sel_idx = (idx_list==i)
        data_matrix_entry = np.zeros((batch_size, data_matrix.shape[1], data_matrix.shape[2]), dtype=np.float32)
        data_matrix_entry[:np.sum(sel_idx), :, :] = data_matrix[sel_idx, :, :]
        if not zero_padding:
            data_matrix_entry[np.sum(sel_idx):, :, :] = data_matrix_entry[0, :, :]
        DIVIDE_BATCH_LIST.append(jnp.array(data_matrix_entry))
    return DIVIDE_BATCH_LIST

def generate_input_loader(data_matrix, idx_list, batch_size=320, zero_padding=True):
    DIVIDE_BATCH_LIST = []
    for i in np.unique(idx_list):
        sel_idx = (idx_list==i)
        data_matrix_entry = np.zeros((batch_size, data_matrix.shape[1]), dtype=np.float32)
        data_matrix_entry[:np.sum(sel_idx), :] = data_matrix[sel_idx, :]
        if not zero_padding:
            data_matrix_entry[np.sum(sel_idx):, :] = data_matrix_entry[0, :]
        DIVIDE_BATCH_LIST.append(jnp.array(data_matrix_entry))
    return DIVIDE_BATCH_LIST    
  
# methods for local models
def get_train_test_sel(driver_ds):
    time_vals = driver_ds.time.values
    if time_vals[1] - time_vals[0] == 1:
        time = np.array([date(2001, 1, 1) + timedelta(days=int(i)) for i in time_vals])
        n_years = np.int64(np.round(len(time) / 365.25))
        train_years = np.round(np.ceil(n_years / 2))
        train_end_date = date(int(time[0].year + train_years), 1, 1)
        train_sel = time<train_end_date
        test_sel = time>=train_end_date
        return train_sel, test_sel
    elif time_vals[1] - time_vals[0] == 7:
        time = np.array([date(2001, 1, 1) + timedelta(days=int(i)) for i in time_vals])
        n_years = np.int64(np.round(len(time) / 52))
        train_years = np.round(np.ceil(n_years / 2))
        train_end_date = date(int(time[0].year + train_years), 1, 1)
        train_sel = time<train_end_date
        test_sel = time>=train_end_date
        return train_sel, test_sel
    elif time_vals[1] - time_vals[0] >=28:
        time = np.array([date(2001, 1, 1) + timedelta(days=int(i)) for i in time_vals])
        n_years = np.int64(np.round(len(time) / 12))
        train_years = np.round(np.ceil(n_years / 2))
        train_end_date = date(int(time[0].year + train_years), 1, 1)
        train_sel = time<train_end_date
        test_sel = time>=train_end_date
        return train_sel, test_sel

def generate_met_matrix(driver_ds, train_sel, test_sel, train_mode=True):
    time = driver_ds.time.values
    t_min = driver_ds.T2M_MIN.values
    t_max = driver_ds.T2M_MAX.values
    rad = driver_ds.SSRD.values
    ca = driver_ds.CO2.values
    doy = driver_ds.DOY.values
    vpd =  driver_ds.VPD.values
    precipitation = driver_ds.TOTAL_PREC.values
    burned_area = np.full(time.shape[0], 0)
    lat = np.full(time.shape, driver_ds.LAT.values)
    delta_t = np.full(time.shape[0], time[1]-time[0])
    t_mean = np.full(time.shape[0], np.mean((t_min[train_sel]+t_max[train_sel])/2))
    mean_precipitation = np.full(time.shape[0], np.mean(precipitation[train_sel]))
    t_norm = ((t_max + t_min)/2 - t_mean) / np.std((t_max[train_sel] + t_min[train_sel])/2)
    rad_norm = (rad - np.mean(rad[train_sel])) / np.std(rad[train_sel])
    vpd_norm = (vpd - np.mean(vpd[train_sel])) / np.std(vpd[train_sel])
    ca_norm = (ca - np.mean(ca[train_sel])) / np.std(ca[train_sel])

    end_of_year_sel = jnp.where(driver_ds.DOY.values==driver_ds.DOY.values[0].item())[0][1:] - 1
    end_of_year_index = np.full(len(driver_ds.DOY.values), False)
    end_of_year_index[end_of_year_sel] = True
    end_of_year_arr = jnp.array(end_of_year_index).astype(np.float32)

    
    met_matrix = jnp.array([time, t_min, t_max, rad, ca, doy, burned_area,
                   vpd, precipitation, lat, delta_t, t_mean, mean_precipitation,
                            t_norm, rad_norm, vpd_norm, ca_norm, end_of_year_arr]).T
    
    if train_mode:
        return met_matrix[train_sel, :]
    else:
        return met_matrix
    
def generate_site_level_target_matrix(driver_ds, train_sel, train_mode=True, reco=False):
    gpp_target = deepcopy(driver_ds.GPP.values)
    gpp_target_mask = np.invert(np.isnan(gpp_target)).astype(np.float32)
    gpp_target[gpp_target_mask==0] = -9999

    nee_target = deepcopy(driver_ds.NBE.values)
    nee_target_mask = np.invert(np.isnan(nee_target)).astype(np.float32)
    nee_target[nee_target_mask==0] = -9999

    et_target = deepcopy(driver_ds.ET.values)
    et_target_mask = np.invert(np.isnan(et_target)).astype(np.float32)
    et_target[et_target_mask==0] = -9999

    lai_target = deepcopy(driver_ds.LAI.values)
    lai_target_mask = np.invert(np.isnan(lai_target)).astype(np.float32)
    lai_target[lai_target_mask==0] = -9999
    
    if reco:
        reco_target = deepcopy(driver_ds.RECO.values)
        reco_target_mask = np.invert(np.isnan(reco_target)).astype(np.float32)
        reco_target[reco_target_mask==0] = -9999

        target_matrix = jnp.stack([gpp_target, gpp_target_mask, nee_target, nee_target_mask,
                                   et_target, et_target_mask, lai_target, lai_target_mask, reco_target, reco_target_mask]).T
    else:
        target_matrix = jnp.stack([gpp_target, gpp_target_mask, nee_target, nee_target_mask,
                               et_target, et_target_mask, lai_target, lai_target_mask]).T
    if train_mode:
        return target_matrix[train_sel, :]
    else:
        return target_matrix

    

def generate_loader_random(data_matrix, batch_size=320):
    if data_matrix.ndim==3:
        return [data_matrix[i*batch_size:(i+1)*batch_size, :, :] for i in range(data_matrix.shape[0] // batch_size)]
    else:
        return [data_matrix[i*batch_size:(i+1)*batch_size, :] for i in range(data_matrix.shape[0] // batch_size)]
    
    
def create_folder_if_not_exists(folder_path: str,
                                verbose: Optional[bool]=False
    ):
    """
    Create a folder if it does not already exist.

    Checks whether the specified folder path exists. If not, the folder
    is created. Optionally prints information about the operation.

    Parameters
    ----------
    folder_path : str
        Path to the folder to create.
    verbose : bool, optional
        If True, print messages indicating whether the folder was created
        or already existed. Default is False.

    Returns
    -------
    None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if verbose:
            print("Created folder: {}".format(folder_path))
    else:
        if verbose:
            print("Folder path exists: {}".format(folder_path))
    return