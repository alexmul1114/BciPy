# Fix targetness in session.json
import json
import click

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

@click.command()
@click.option('--directory', prompt='Provide a path to a collection of data folders to be updated to BciPy 2.0',
              help='The path to data')
def process_data(directory):
    session_name = f'{directory}/session.json'
    data = json.load(open(session_name))
    update_epochs(data)
    with open(session_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

# TODO Fix targetness in triggers.

if __name__ == '__main__':
    process_data()



# Notes for Dan. These are conversion compatiable, can be loaded into 2.0.. targetness labels are fixed.. json is not 100% 2.0 but the information in them is now correct. It still uses stuff like epochs instead of series. We should indicate it was collected on whatever version and upgraded to allow for loading in newer versions. The json would always have to be manually processed so nothing breaks either way but you may want to document what an epoch is and how that file breaks down to inquiries. 