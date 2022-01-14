import logging
import mne
import numpy as np
from mne import Annotations, Epochs
from mne.io import RawArray
from pathlib import Path
from typing import Tuple
from bcipy.helpers.acquisition import analysis_channel_names_by_pos, analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.stimuli import play_sound
from bcipy.helpers.system_utils import report_execution_time
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.visualization import visualize_erp
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.process import get_default_transform

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s')

@report_execution_time
def ar_offline_analysis(data_folder: str = None,
                        parameters: dict = {},
                        alert_finished: bool = True):
    if not data_folder:
        data_folder = load_experimental_data()

    # extract relevant session information from parameters file
    trial_length = parameters.get('trial_length')
    trials_per_inquiry = parameters.get('stim_length')
    triggers_file = parameters.get('trigger_file_name', 'triggers.txt')
    raw_data_file = parameters.get('raw_data_name', 'raw_data.csv')

    # get signal filtering information
    downsample_rate = parameters.get('down_sampling_rate', 2)
    notch_filter = parameters.get('notch_filter_frequency', 60)
    hp_filter = parameters.get('filter_high', 45)
    lp_filter = parameters.get('filter_low', 2)
    filter_order = parameters.get('filter_order', 2)

    # get offset and k folds
    static_offset = 0
    k_folds = parameters.get('k_folds', 10)

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    fs = raw_data.sample_rate

    log.info(f'Channels read from csv: {channels}')
    log.info(f'Device type: {type_amp}')

    default_transform = get_default_transform(
        sample_rate_hz=fs,
        notch_freq_hz=notch_filter,
        bandpass_low=lp_filter,
        bandpass_high=hp_filter,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    data, fs = default_transform(raw_data.by_channel(), fs)

    trigger_values, trigger_timing, _ = trigger_decoder(
        offset=static_offset,
        trigger_path=f'{data_folder}/{triggers_file}.txt')

    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, type_amp)
    channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
    data = np.delete(data, channels_to_remove, axis=0)
    channels = np.delete(channels, channels_to_remove, axis=0).tolist()
    import pdb; pdb.set_trace()


    mne_pipeline(data, channels, fs, trigger_values, trigger_timing, trial_length=trial_length)



""" MNE Exploration goals

1. can we load data into MNE? What are the pieces of data we need? 
    a. YES
    b. We need data by channel, channel information, sample rate
    c. We need trigger labels, duration and timing in seconds (for epoching)
2. can we plot data in MNE?
    a. YES, some plots are limited to those with Montages.
3. can we process data using MNE? ICA? Other techniques?
4. can we generate reports using mne?
5. can we use this to do a min / max evaluation?
    a. YES... we should remove what we can in our signal module and try abstracting mne api in evaluator
6. We should set an EEG reference channel
7. Compare MNE and BciPy filtering

Interested most in processing and reporting potential...

It creates a baseline epoch for you! and requires it through... we don't baseline correct now and it may be a good change to add it. We can go back and forth between 
    mne RawData and numpy which is needed for our signal model.


Montages and location estimates fundamental to many estimates in EEG processing.
We can describe a default and mapping to those channels.... but we should allow some custom montages? Or at least support for defining them.

+ adding locations https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html#working-with-built-in-montages

+ cartesion coordinates (p1, p2, p3) and digital coordinates. see https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_montage, necessary especially for ECoG.
"""

def mne_pipeline(data, channels, fs, trigger_labels, trigger_timing, trial_length=0.5):
    # define a standard montage for electrode locations. There are more defaults
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    channel_types = ['eeg'] * len(channels)
    info = mne.create_info(channels, fs, channel_types)
    mne_data = RawArray(data, info)
    mne_data.set_montage(ten_twenty_montage)
    annotations = Annotations(trigger_timing, [trial_length] * len(trigger_timing), trigger_labels)
    mne_data.set_annotations(annotations)
    # mne_data.plot()

    events_from_annot, event_dict = mne.events_from_annotations(mne_data)
    epochs = Epochs(mne_data, events_from_annot)
    nontarget = epochs['1']
    target = epochs['2']

    # https://mne.tools/stable/auto_tutorials/epochs/20_visualize_epochs.html

    # from here you can plot the epoched data in all or a subset of channels with plot
    # target.plot()
    # target.plot_psd()
    # target.plot_image()
    nontarget_average = nontarget.average()
    target_average = target.average()

    # target_average.plot_joint(times=[-0.2, 0,  0.25, 0.5]) # uncomment to plot a joint topomap and eeg plot

    # Creating an evoked data structure which has more plotting tools https://mne.tools/stable/auto_tutorials/evoked/10_evoked_overview.html
    # https://mne.tools/stable/auto_tutorials/evoked/20_visualize_evoked.html
    # evokeds = dict(auditory=list(epochs['auditory/left'].iter_evoked()),
    #            visual=list(epochs['visual/left'].iter_evoked()))
    # mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks=picks)
    import pdb; pdb.set_trace()
    


def semi_automatic_artifact_rejection():
    """You can find events manually using plot, then using reject_by_annotation
    
    You can find event programmatically https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html#tut-artifact-overview
    """
    # https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html
    pass

def eye_artifacts(mne_data):
    #  Use this to create Epochs out of the eog events found
    # eog_epochs = mne.preprocessing.create_eog_epochs(mne_data, baseline=(-0.2, 0.0), picks=['Fp1', 'Fp2'])
    # eog_epochs.plot_image(combine='mean')
    # eog_epochs.average().plot_joint()

    # How to find blinks in the channel fp1 and set them to be filtered later with reject_by_annotation
    #  See https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html
    eog_events = mne.preprocessing.find_eog_events(mne_data, ch_name='Fp1')

    onsets = eog_events[:, 0] / mne_data.info['sfreq'] - 0.25
    durations = [0.5] * len(eog_events)
    descriptions = ['bad blink'] * len(eog_events)
    blink_annot = mne.Annotations(onsets, durations, descriptions,
                                orig_time=mne_data.info['meas_date'])
    mne_data.set_annotations(blink_annot)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default=None)
    parser.add_argument('-p', '--parameters_file',
                        default='bcipy/parameters/parameters.json')
    args = parser.parse_args()

    log.info(f'Loading params from {args.parameters_file}')
    parameters = load_json_parameters(args.parameters_file,
                                      value_cast=True)
    ar_offline_analysis(args.data_folder, parameters)
    log.info('Offline Analysis complete.')