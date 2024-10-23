import matplotlib.pyplot as plt
import numpy as np

def plot_site_figure(driver_ds, output_matrix_full, train_sel, test_sel, nnse_eval, save_name, reco=False):
    gpp_tr, nee_reco_tr, et_tr, lai_tr, gpp_te, nee_reco_te, et_te, lai_te = nnse_eval
    fig, axs = plt.subplots(4,1, figsize=(10,10), dpi=300)
    ax = axs.flatten()
    ax[0].plot(driver_ds.GPP.values[train_sel], alpha=0.5)
    ax[0].plot(np.arange(np.sum(train_sel), len(test_sel)), driver_ds.GPP.values[test_sel], alpha=0.5)
    ax[0].plot(output_matrix_full[:, 1])
    ax[0].set_ylabel("GPP")
    ax[0].text(100, np.nanmax(driver_ds.GPP.values)*0.9, "Train: {:.3f}".format(gpp_tr))
    ax[0].text(len(driver_ds.GPP.values) * 0.7, np.nanmax(driver_ds.GPP.values)*0.9, "Test: {:.3f}".format(gpp_te))

    if not reco:
        ax[1].plot(driver_ds.NBE.values[train_sel], alpha=0.5)
        ax[1].plot(np.arange(np.sum(train_sel), len(test_sel)), driver_ds.NBE.values[test_sel], alpha=0.5)
        ax[1].plot(output_matrix_full[:, 21])
        ax[1].set_ylabel("NEE")
        ax[1].text(100, np.nanmax(driver_ds.NBE.values)*0.9, "Train: {:.3f}".format(nee_reco_tr))
        ax[1].text(len(driver_ds.NBE.values) * 0.7, np.nanmax(driver_ds.NBE.values)*0.9, "Test: {:.3f}".format(nee_reco_te))
    else:
        ax[1].plot(driver_ds.RECO.values[train_sel], alpha=0.5)
        ax[1].plot(np.arange(np.sum(train_sel), len(test_sel)), driver_ds.RECO.values[test_sel], alpha=0.5)
        ax[1].set_ylabel("RECO")
        ax[1].plot(output_matrix_full[:, 21] + output_matrix_full[:, 1])
        ax[1].text(100, np.nanmax(driver_ds.RECO.values)*0.9, "Train: {:.3f}".format(nee_reco_tr))
        ax[1].text(len(driver_ds.RECO.values) * 0.7, np.nanmax(driver_ds.RECO.values)*0.9, "Test: {:.3f}".format(nee_reco_te))
       
    ax[2].plot(driver_ds.ET.values[train_sel], alpha=0.5)
    ax[2].plot(np.arange(np.sum(train_sel), len(test_sel)), driver_ds.ET.values[test_sel], alpha=0.5)
    ax[2].plot(output_matrix_full[:, 2])
    ax[2].set_ylabel("ET")
    ax[2].text(100, np.nanmax(driver_ds.ET.values)*0.9, "Train: {:.3f}".format(et_tr))
    ax[2].text(len(driver_ds.ET.values) * 0.7, np.nanmax(driver_ds.ET.values)*0.9, "Test: {:.3f}".format(et_te))

    ax[3].plot(driver_ds.LAI.values[train_sel], alpha=0.5)
    ax[3].plot(np.arange(np.sum(train_sel), len(test_sel)), driver_ds.LAI.values[test_sel], alpha=0.5)
    ax[3].plot(output_matrix_full[:, 0])
    ax[3].set_ylabel("LAI")
    ax[3].text(100, np.nanmax(driver_ds.LAI.values)*0.9, "Train: {:.3f}".format(lai_tr))
    ax[3].text(len(driver_ds.LAI.values) * 0.7, np.nanmax(driver_ds.LAI.values)*0.9, "Test: {:.3f}".format(lai_te))
    plt.savefig(save_name)