from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/caoyu/CODE/pytracking2/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/home/caoyu/Dataset/OTB100'
    settings.result_plot_path = '/home/caoyu/CODE/pytracking2/pytracking/pytracking/result_plots/'
    settings.results_path = '/home/caoyu/CODE/pytracking2/pytracking/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/caoyu/pytracking2/CODE/pytracking/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings
