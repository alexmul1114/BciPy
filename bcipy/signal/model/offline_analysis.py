import logging
import mne
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
from matplotlib.figure import Figure
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
def offline_analysis(data_folder: str = None,
                     parameters: dict = {}, alert_finished: bool = True) -> Tuple[SignalModel, Figure]:
    """ Gets calibration data and trains the model in an offline fashion.
        pickle dumps the model into a .pkl folder
        Args:
            data_folder(str): folder of the data
                save all information and load all from this folder
            parameter(dict): parameters for running offline analysis
            alert_finished(bool): whether or not to alert the user offline analysis complete

        How it Works:
        - reads data and information from a .csv calibration file
        - reads trigger information from a .txt trigger file
        - filters data
        - reshapes and labels the data for the training procedure
        - fits the model to the data
            - uses cross validation to select parameters
            - based on the parameters, trains system using all the data
        - pickle dumps model into .pkl file
        - generates and saves ERP figure
        - [optional] alert the user finished processing
    """

    if not data_folder:
        data_folder = load_experimental_data()

    # extract relevant session information from parameters file
    trial_length = parameters.get('trial_length')
    trials_per_inquiry = parameters.get('stim_length')
    triggers_file = parameters.get('trigger_file_name', 'triggers')
    raw_data_file = parameters.get('raw_data_name', 'raw_data.csv')

    # get signal filtering information
    downsample_rate = parameters.get('down_sampling_rate')
    notch_filter = parameters.get('notch_filter_frequency')
    hp_filter = parameters.get('filter_high')
    lp_filter = parameters.get('filter_low')
    filter_order = parameters.get('filter_order')

    # get offset and k folds
    static_offset = parameters.get('static_trigger_offset', 0.0)
    k_folds = parameters.get('k_folds')

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

    # Process triggers.txt
    trigger_values, trigger_timing, _ = trigger_decoder(
        offset=static_offset,
        trigger_path=f'{data_folder}/{triggers_file}.txt')

    import pdb; pdb.set_trace()

    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, type_amp)
    # channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
    # data = np.delete(data, channels_to_remove, axis=0)
    # mne_pipeline(data, channels, fs, t_t_i, t_i, trial_length)

    model = PcaRdaKdeModel(k_folds=k_folds)
    data, labels = model.reshaper(
        trial_labels=trigger_values,
        timing_info=trigger_timing,
        eeg_data=data,
        fs=fs,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        trial_length=trial_length)

    log.info('Training model. This will take some time...')
    model.fit(data, labels)
    model_performance = model.evaluate(data, labels)

    log.info(f'Training complete [AUC={model_performance.auc:0.4f}]. Saving data...')


    model.save(data_folder + f'/model_{model_performance.auc:0.4f}.pkl')

    visualize_erp(
        data,
        labels,
        fs,
        plot_average=False,  # set to True to see all channels target/nontarget
        save_path=data_folder,
        channel_names=analysis_channel_names_by_pos(channels, channel_map),
        show_figure=False,
        figure_name='average_erp.pdf'
    )
    # if alert_finished:
    #     offline_analysis_tone = parameters.get('offline_analysis_tone')
    #     play_sound(offline_analysis_tone)

    # import pdb; pdb.set_trace()
    return


# def _remove_bad_data_by_trial(trial_data, trial_labels, parameters,estimate):

#     """ Removes bad data in a trial-by-trial fashion. Offline artifact rejection.
#         Args:
#         trial_data: a multidimensional array of Channels x Trials (chopped into 500ms chunks) x Voltages 
#         trial_labels: an ndarray of 0s (non-targets) and 1s (target), each representing a trial
#         parameter(dict): parameters to pull information for enabled rules and threshold values

#      """
#     # get enabled rules
#     high_voltage_enabled = parameters['high_voltage_threshold']
#     low_voltage_enabled = parameters['low_voltage_threshold']
#     rejection_threshold = parameters['rejection_threshold']

#     # invoke evaluator / rules
#     evaluator = Evaluator(parameters, high_voltage_enabled, low_voltage_enabled)

#     # iterate over trial data, evaluate the trials, remove if needed and modify the trial labels to reflect
#     channel_number = trial_data.shape[0]
#     trial_number = trial_data[0].shape[0]
#     num_trials = trial_data[0].shape[0]

#     trial = 0
#     rejected_trials = 0
#     rejection_suggestions = 0
#     bad_channel_threshold = 1

#     while trial < trial_number:
#         # go channel-wise through trials
#         for ch in range(channel_number):
#             data = trial_data[ch][trial]
#             # evaluate voltage samples from this trial
#             response = evaluator.evaluate(data) 
#             if not response: # if False
#                 rejection_suggestions += 1 
#                 if rejection_suggestions >= bad_channel_threshold:
#                     # if the evaluator rejects the data and we've reached
#                     # the threshold, then delete the trial from each channel,
#                     # adjust trial labels to follow suit, then exit the loop
#                     trial_data = np.delete(trial_data, trial, axis=1)
#                     trial_labels = np.delete(trial_labels, trial)
#                     trial_number -= 1
#                     rejected_trials += 1
#                     break
#         rejection_suggestions = 0 
#         trial += 1

#     percent_rejected = (rejected_trials / num_trials) * 100

#     print('Number Rejected Trial-based: ' + str(rejected_trials))

#     if percent_rejected > rejection_threshold:

#         raise Exception(f'Percentage of data removed too high [{percent_rejected}]')

#     else:
#         return trial_data, trial_labels, percent_rejected

# def _remove_bad_data_by_sequence(trial_data, trial_labels, parameters, trials_per_sequence,estimate):
#     """Remove Bad Data By Sequence.
# ​
#     Removes bad data in a sequence-by-sequence fashion. Offline artifact rejection.
#         Args:
#         trial_data: a multidimensional array of Channels x Trials (chopped into 500ms chunks) x Voltages
#         trial_labels: an ndarray of 0s (non-targets) and 1s (target), each representing a trial
#         parameter(dict): parameters to pull information for enabled rules and threshold values

#     """

#     # get enabled rules
#     high_voltage_enabled = parameters['high_voltage_threshold']
#     low_voltage_enabled = parameters['low_voltage_threshold']
#     rejection_threshold = parameters['rejection_threshold']

#     # invoke evaluator / rules
#     evaluator = Evaluator(parameters, high_voltage_enabled, low_voltage_enabled)

#     # iterate over trial data, evaluate the sequences, remove if needed and modify the trial labels to reflect
#     channel_number = trial_data.shape[0]

#     # looping criteria changes by number of trials in a sequence rather than 1
#     # which is used in _remove_bad_data_by_trial
#     trial_number = trial_data[0].shape[0]
#     num_trials = trial_data[0].shape[0]

#     trial = 0
#     rejected_trials = 0

#     while trial < trial_number:
#         # go channel-wise through trials
#         for ch in range(channel_number):
#             sequence_end = trial + trials_per_sequence
#             data = trial_data[ch][trial:sequence_end]
#             # evaluate voltage samples from this sequence
#             # by iterating through trials within the sequence
#             response = evaluator.evaluate(data)
#             if not response:
#                 # if the evaluator rejects the data and we've reached
#                 # the threshold, then delete the entire sequence from each channel,
#                 # adjust trial labels to follow suit, then exit the loop
#                 trial_data = np.delete(trial_data, np.s_[trial:sequence_end], axis=1)
#                 trial_labels = np.delete(trial_labels, np.s_[trial:sequence_end])
#                 trial_number -= trials_per_sequence
#                 rejected_trials += trials_per_sequence
#                 break

#         rejection_suggestions = 0
#         trial += trials_per_sequence

#     percent_rejected = (rejected_trials / num_trials ) * 100

#     print('Number Rejected Sequence-based: ' + str(rejected_trials))

#     if percent_rejected > rejection_threshold:

#         raise Exception(f'Percentage of data removed too high [{percent_rejected}]')

#     else:
#         return trial_data, trial_labels, percent_rejected
    
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
    offline_analysis(args.data_folder, parameters)
    log.info('Offline Analysis complete.')
