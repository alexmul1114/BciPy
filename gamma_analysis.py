"""
Transforms needed to do Gamma Analysis on Tripolar Electrodes.

Written By. Tab Memmott, tab.memmott@gmail.com

Last updated 06/01/2020 #BLM

NTS - Exports done, looking to see if I can create my own averages by changing the triggers I send back and using meta_calculation.
    This method likely needs to flatten the structure to compute any averages, etc
"""
import csv
import numpy as np
import pandas as pd

from bcipy.helpers.load import read_data_csv, load_json_parameters
from bcipy.signal.process.filter import bandpass, notch, downsample
from bcipy.helpers.task import trial_reshaper
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.acquisition import (
    analysis_channels, analysis_channel_names_by_pos)
from bcipy.signal.process.decomposition.psd import (
    power_spectral_density, PSD_TYPE)

# defaults and constants
DOWNSAMPLE_RATE = 1
NOTCH_FREQ = 60
FILTER_HP = 1
FILTER_LP = 80  # 50 alpha 80 gamma
FILTER_ORDER = 8
TRIPOLAR_CHANNELS = [0, 2, 4, 6]
TRIGGERS_OF_INTEREST = ['first_pres_target']
# TRIGGERS_OF_INTEREST = ['nontarget']

FREQ_RANGE = [50, 70]

SUBJECT = 'jk'
BAND = 'upper_gamma'
TASK = 'eyes'
FREQ_INTERVAL = '50_70'
TRIGGERS_USED = 'first_pres_target'
CONDITION = 'first_half'

CSV_EXPORT_NAME = f'tripolar_analysis_{BAND}_{FREQ_INTERVAL}_{TASK}_{TRIGGERS_USED}_{CONDITION}_{SUBJECT}.csv'
META_OUTPUT_NAME = f'tripolar_analysis_{BAND}_{FREQ_INTERVAL}_{TASK}_average_{TRIGGERS_USED}_{CONDITION}_{SUBJECT}.txt'
# EXPORT_CHANNELS = [0]

def calculate_fft(data, fs, trial_length, relative=False, plot=False):
    """Calculate FFT Gamma

    Calculate the amount of gamma using FFT.
    """
    return power_spectral_density(
                data,
                FREQ_RANGE,
                sampling_rate=fs,
                window_length=trial_length,
                method=PSD_TYPE.WELCH,
                plot=plot,
                relative=relative)


def get_experiment_data(raw_data_path, parameters_path, apply_filters=False, scale=False, scale_factor=5, scale_channels=[1, 3, 5, 7]):
    """Get Experimental Data.

    Given the path to raw data and parameters, parse them into formats we can
        work with. Optionally apply the default filters to the data.
        To change filtering update the constants at the top of file or do custom
        tranforms after the if apply_filters check.
    """
    raw_data, _, channels, type_amp, fs = read_data_csv(raw_data_path)
    parameters = load_json_parameters(parameters_path,
                                      value_cast=True)

    if apply_filters:
        # filter the data as desired here!
        raw_data, fs = filter_data(
            raw_data, fs, DOWNSAMPLE_RATE, NOTCH_FREQ)

    if scale:
        raw_data = scale_data(raw_data, scale_factor, scale_channels)
    return raw_data, channels, type_amp, fs, parameters


def scale_data(raw_data, scale_factor, scale_channels):
    """Scale Data.
    
    Using the scale channels defined, scale the raw_data channels corresponding to that index by the scale_factor.
    """
    def func(x):
        for index in scale_channels:
            x[index] = x[index] / scale_factor

    np.apply_along_axis(func, 0, raw_data)
    return raw_data


def get_triggers(trigger_path, poststim, prestim=False):
    # decode triggers
    _, trigger_targetness, trigger_timing, offset = trigger_decoder(
        mode='calibration',
        trigger_path=trigger_path,
        triggers=TRIGGERS_OF_INTEREST)

    # prestim must be a positive number. Transform the trigger timing if
    #   a prestimulus amount is wanted. Factor that into the trial length for
    #   later reshaping
    if prestim and abs(prestim) > 0:
        trigger_timing = transform_trigger_timing(trigger_timing, prestim)
        trial_length = poststim + abs(prestim)
    else:
        trial_length = poststim

    
    # # Remove the last stimuli if more buffer is needed post stim
    # trigger_timing.pop()
    # trigger_targetness.pop()

    # Take only the first half
    length = int(len(trigger_timing) / 2)
    trigger_timing = trigger_timing[:length]
    trigger_targetness = trigger_targetness[:length]
    # Take only the last
    # length = len(trigger_timing) / 2
    # trigger_timing.pop()
    # trigger_targetness.pop()
    # trigger_timing = trigger_timing[length:]
    # trigger_targetness = trigger_targetness[length:]

    return trigger_timing, trigger_targetness, offset, trial_length


def transform_trigger_timing(trigger_timing, pre_stim):
    """Transform Trigger Timing.

    Given a list of times and a prestimulus amount, shift every
        item in the array by that amount and return the new triggers.

    Note. Given pre_stim is in ms and triggers are in seconds, we 
        transform that here.
    """
    new_triggers = []
    for trigger in trigger_timing:
        new_triggers.append(trigger - pre_stim)

    return new_triggers


def filter_data(raw_data, fs, downsample_rate, notch_filter_freqency):
    """Filter Data.
    Using the same procedure as AD supplement, filter and downsample the data
        for futher processing.
    Return: Filtered data & sampling rate
    """
    notch_filterted_data = notch.notch_filter(
        raw_data, fs, notch_filter_freqency)
    bandpass_filtered_data = bandpass.butter_bandpass_filter(
        notch_filterted_data, FILTER_HP, FILTER_LP, fs, order=FILTER_ORDER)
    filtered_data = downsample.downsample(
        bandpass_filtered_data, factor=downsample_rate)
    sampling_rate_post_filter = fs / downsample_rate
    return filtered_data, sampling_rate_post_filter


def parse(
        data,
        fs,
        channels,
        type_amp,
        triggers,
        targetness,
        offset,
        trial_length,
        parameters):
    """Parse.

    Using the data collected via BciPy, reshape and return
       the parsed data and labels (target/nontarget)
    """

    # add a static offset of the system.
    # This is calculated on a per machine basis.
    # Reach out if you have questions.
    offset = offset + .067 # fuck

    # reshape the data! *Note* to change the channels you'd like returned
    # from the reshaping, create a custom channel map and pass it into
    # the named arg channel_map. It is a list [0, 1, 0, 1], where 1 is
    # a channel to keep and 0 is to remove. Must be the same length as 
    # channels.
    trials, labels, _, _ = trial_reshaper(
        targetness,
        triggers,
        data,
        mode='free_spell', # first_pres_target needs this to be free_spell. All others can use calibration
        fs=fs,
        k=1,
        offset=offset,
        channel_map=analysis_channels(channels, type_amp),
        trial_length=trial_length)

    return trials, labels


def export_data_to_csv(path, exports, intervals, targetness, channels):
    """Export Data to CSV.
    
    Given an array of exports and column names, write a csv for processing in other systems
    """
    interval_len = len(intervals)
    with open(f'{path}/{CSV_EXPORT_NAME}', 'w') as tripolar_export:
        writer = csv.writer(
            tripolar_export,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)

        headers = ['targetness']

        for channel in channels:
            for interval in intervals:
                first = f'neg{abs(interval[0])}' if interval[0] < 0 else interval[0]
                second = f'neg{abs(interval[1])}' if interval[1] < 0 else interval[1]
                headers.append(f'{channel}_interval_{first}_{second}')
        # write headers
        writer.writerow(headers)

        # write PSD data
        i = 0
        data = []
        for trial in range(len(targetness)):
            row = [targetness[i]]

            # loop over channels
            for channel in range(len(exports)):

                for idx in range(interval_len):
                    row.append(exports[channel][trial][idx])
            
            writer.writerow(row)
            row.pop(0)
            data.append(row) # Create an export for calculation
            i += 1
    return data

def meta_calculation(path, data):
    """Meta Calculation.
    
    Calculate the average of all channels and save as text
    """
    reporting = []

    array = np.array(data)
    col_average = np.mean(array, axis=0)

    prestim_intervals = 1
    post_stim_intervals = 4
    y = 0 # keep track of column
    while len(col_average) > y:
        baseline_average = col_average[y]

        y += prestim_intervals
        for _ in range(post_stim_intervals):
            
            # baseline correct using the prestim average
            reporting.append(col_average[y] - baseline_average)
            y += 1

    np.savetxt(f'{path}/{META_OUTPUT_NAME}', col_average)

    with open(f'{path}/baseline_corrected_{META_OUTPUT_NAME}', 'w') as output:
        for item in reporting:
            output.write('%s\n' % item)

    return reporting, col_average


def determine_export_bins(data_export_range, interval):
    """Determine export bins.
    
    assumes data range value 1 less than 2.
    """
    # determine the range of our two values 
    diff = abs(data_export_range[0] - data_export_range[1])

    intervals = []

    # start with the smallest number and add interval to it
    j = data_export_range[0]
    for _ in range( int(diff / interval)):
        intervals.append([j, j + interval])
        j += interval

    return intervals


def generate_interval_trials(trials, interval, intervals, fs, channel):
    """Generate Interval Trials.
    
    Using the trialed data from the trial reshaper, break the data into interval for export
    """
    export = {}

    # convert interval to ms and calculate samples
    samples_per_interval = int((interval) * fs )
    i = 0
    for trial in trials[channel]:
        j = 0
        k = samples_per_interval
        z = 0
        export[i] = {}
        for _ in intervals:
            # export[i][z] = trial[j:k]
            export[i][z] = calculate_fft(trial[j:k], fs, interval)

            j = k
            k += samples_per_interval
            z += 1

        i += 1

    return export


if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format='(%(threadName)-9s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-pre', '--prestim', default=.25, type=int)
    parser.add_argument('-post', '--poststim', default=1, type=int)
    parser.add_argument('-int', '--interval', default=.25, type=int)
    args = parser.parse_args()

    # extract relevant args
    data_folder = args.path
    prestim = args.prestim
    poststim = args.poststim
    interval = args.interval

    intervals = determine_export_bins([-prestim, poststim], interval)

    raw_data_path = '{}/raw_data.csv'.format(data_folder)
    parameter_path = '{}/parameters.json'.format(data_folder)
    trigger_path = '{}/triggers.txt'.format(data_folder)

    logging.info('Reading information from {}, with prestim=[{}] and poststim=[{}]'.format(
        data_folder, prestim, poststim))

    logging.info('Reading EEG data \n')
    # get the experiment data needed for processing.
    # *Note* Constants for filters at top of file. Set apply_filters to true to use the filters.
    data, channels, type_amp, fs, parameters = get_experiment_data(
        raw_data_path, parameter_path, apply_filters=True,
        scale=True, scale_factor=5, scale_channels=TRIPOLAR_CHANNELS) # Here we scale the tripolar channels because the gain is higher on them

    logging.info('Reading trigger data \n')
    # give it path and poststimulus length (required). Last, prestimulus length (optional)
    triggers, targetness, offset, trial_length = get_triggers(trigger_path, poststim, prestim=prestim)

    # Is trial length supposed to be in ms or seconds?

    logging.info(f'Parsing into trials of trial length {trial_length} \n')
    # parse the data and return the trials and labels to use with DWT
    trials, labels = parse(
        data,
        fs,
        channels,
        type_amp,
        triggers,
        targetness,
        offset,
        trial_length,
        parameters)
    
    exports = []
    for channel in range(len(channels)):
        exports.append(
            generate_interval_trials(trials, interval, intervals, fs, channel=channel)
        )

    # Do your analyses or uncomment next line to use debugger here and see what is returned.
    # Exports is an array of channel name and export information
    data = export_data_to_csv(data_folder, exports, intervals, targetness, channels)
    # WE ARE AT THE POINT OF ME BASELINE CORRECTING THE AVERAGES TO USE FOR PLOTTING AND EXPLORATION
    reporting, col_average = meta_calculation(data_folder, data) # Uncomment to save average of columns

    # calculate_fft(trials[0][1], fs, trial_length, plot=True)
    logging.info('Complete! \n')
