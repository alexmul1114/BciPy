import imp
import click
from bcipy.helpers.load import fast_scandir
from bcipy.helpers.convert import convert_to_edf
import json

def update_epochs(data) -> list:
    series = data['epochs']

    for inquiries in series.values():
        for inquiry in inquiries.values():
            copy_phrase = inquiry['copy_phrase']
            current_text = inquiry['current_text']
            if copy_phrase[0:len(current_text)] == current_text:
            # if correctly spelled so far, get the next letter.
                target_letter = copy_phrase[len(current_text)]
            else:
                target_letter = '<'

            try:
                target_index = inquiry['stimuli'][0].index(target_letter)
            except ValueError:
                target_index = None

            inquiry['target_info'][0] = 'fixation'
            if target_index:
                inquiry['target_info'][target_index] = 'target'
    return series

def get_targetness(data):

    targetness = []
    series = data['epochs']

    for inquiries in series.values():
        for inquiry in inquiries.values():
            targetness.extend(inquiry['target_info'])
    return targetness


def process_data(directory):
    session_name = f'{directory}/session.json'
    try:
        data = json.load(open(session_name))
        targetness = get_targetness(data)
        # with open(session_name, 'w', encoding='utf-8') as json_file:
        #     json.dump(data, json_file, ensure_ascii=False, indent=2)
        return targetness
    except FileNotFoundError:
        print(f'skipping {directory}')



@click.command()
@click.option('--directory', prompt='Provide a path to a collection of data folders to be updated to BciPy 2.0',
              help='The path to data')
@click.option('--convert', '-c', is_flag=True, help="Print more output.")
def update_triggers(directory, convert):
    """Update Triggers."""

    particpants = fast_scandir(directory)
    for part_dir in particpants:
        eeg_level_dir = fast_scandir(part_dir)
        for top_level in eeg_level_dir:
            top_level_session_dirs = fast_scandir(top_level)
            for top_session in top_level_session_dirs:
                sessions = fast_scandir(top_session)
                for session in sessions:
                    # trigger_file = f'{session}\\triggers.txt'
                    # triggers = load_triggers(trigger_file)
                    # test = correct_triggers(triggers)
                    # write_triggers(test, path=trigger_file)

                    # targetness = process_data(session)

                    # if targetness:
                    #     corrected_triggers = correct_targetness_triggers(triggers, targetness)
                    #     write_triggers(corrected_triggers, path=trigger_file)

                    # if convert:
                    try:
                        edf_path = convert_to_edf(
                            session,
                            use_event_durations=False,
                            write_targetness=True,
                            overwrite=True,
                            annotation_channels=4)
                    except:
                        print(f'skipping session {session}')

def load_triggers(trigger_path):
    # Get every line of triggers.txt
    with open(trigger_path, 'r+') as text_file:
        trigger_txt = [line.split() for line in text_file]
    return trigger_txt

# def correct_triggers(triggers):
#     calibration_trigger = triggers.pop(0)
#     offset_trigger = triggers.pop(-1)

#     new_offset_value = float(offset_trigger[-1]) - float(calibration_trigger[-1])
#     offset = ['starting_offset', 'offset', new_offset_value]

#     new_triggers = [offset]
#     for trigger in triggers:
#         label = trigger[0]
#         targetness = trigger[1]
#         time = trigger[2]
#         if targetness == 'first_pres_target':
#             targetness = 'prompt'
#         new_triggers.append([label, targetness, time])
#     return new_triggers

def write_triggers(triggers, path='triggers_update.txt'):
    with open(path, 'w+', encoding='utf-8') as trigger_write:
        for trigger in triggers:
            trigger_write.write(f'{trigger[0]} {trigger[1]} {trigger[2]}\n')


def correct_targetness_triggers(triggers, targetness):

    offset = triggers.pop(0)
    new_triggers = [offset]
    for label, trigger in zip(targetness, triggers):
        trigger[1] = label
        new_triggers.append(trigger)

    return new_triggers
            



if __name__ == '__main__':
    update_triggers()