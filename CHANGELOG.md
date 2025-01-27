# 2.0.0-rc.1

## Contributions

This version contains major refactoring efforts and features. We anticipate a few additional refactor efforts in the near term based on feature requests from the community and (CAMBI)[cambi.tech]. These will support multi-modality, data sharing, and more complex language modeling. We are utilizing a release candidate to make features and bugfixes available sooner despite the full second version being in-progress. Thank you for your understanding and continued support! All pull requests in Github from #123 until #217 represent the 2.0.0-rc.1 work. 

The highlights: 

- `Acquisition Enhancements`: multi-modal support and better performance overall! #171, #174
- `Language Model Refactor`: deprecation of docker base models. Addition of `LanguageModel` base class, `UniformLanguageModel` and a hugggingface model `GPT2LanguageModel`. #207
- `Signal Model Refactor`: Refactor with base class definitions and general cleanup. PcaRdaKde model updates to decrease training time and limit the magnitude of likelihood responses. #132, #140, #141, #147, #208
- `Matrix Display: SCP`: A single character flash Matrix speller is now integrated. A `MatrixDisplay` and accompanying `MatrixCalibration` + `Matrix Time Test Calibration`. #192, #205, #213
- `Inquiry Preview`: a mode in RSVP spelling that allows a user to see an inquiry before it is presented in a serial fashion. The user may also engage with the preview using a key press; either to confirm or skip an inquiry. See below for more details.
- `GUI updates`: all BciPy core GUIs and methods are converted to PyQt5 for better cross-platform availability and tooling. See below for more details.
- `Linux compatibility`: with the upgrading of dependencies and a helpful shell script (see `scripts/shell/linux_requirements.sh`) for setting up new machines, we are linux compatible. See below for more details.
- `Prestimulus buffer and Inquiry Based Training` - support to add prestimulus data to reshaping and data queries to permit better filter application. Additionally, use this buffer and the inquiry reshaper to mimic data experienced in real time during training in offline_analysis.py #208

The details (incomplete, our apologies!):

### Added

- `Makefile`: `run-with-defaults` make command for running `bcipy` and `viewer` command for running the data viewer #149
- `.bcipy/README.md`: describes experiments and fields in greater detail #156
- `signal/process/filter.py`: add `Notch` and `Bandpass` as classes #147
- `signal/process/transform.py`: add `Composition`, `Downsample`, and `get_default_transform` #147
- `helpers/visualization.py`: moved plot_edf function from demo to this module #126
- `helpers/raw_data.py`: module for raw data format support in BciPy #160
- `helpers/load.py`: add `load_users` method for extracting user names from data save location #125 add extract_mode method for determining the mode #126
- `helpers/task/`: add `Reshaper`, refactor trial reshaper and add inquiry reshaper #147 add `get_key_press` method with custom stamp argument. #129 
- `helpers/stimuli.StimuliOrder`: defined ordering of inquiry stimuli. The current approach is to randomize. This adds an alphabetical option. #153 
- `helpers/stimuli.alphabetize`: method for taking a list of strings and returning them in alphabetical order with characters last in the list. #153
- `helpers/validate`: `_validate_experiment_fields` and `validate_experiments`: validates experiments and fields in the correct format #156
- `bcipy/helpers/system_utils`: add report execution decorator #163
- `scripts/shell/linux_requirements.sh`: add script for installing necessary dependencies on linux systems #167
- `.github/workflows/main.yml`: adds support for CI in BciPy #166
- `bcipy/gui/file_dialog.py`: PyQt5 window for prompting for file locations #168
- `display/paradigm/matrix`: added MatrixDisplay class with single-character presentation (SCP). #180

### Updated

- `LICENSE.md`: to used the Hippocratic license 2.1
- `CODE_OF_CONDUCT.md`: to latest version of the Contributor Covenant
- `README.md`: Add new glossary terms: mode, session and task #126 #127 and cleanup #129
- `bcipy/main.py`: formally, `bci_main.py`. To give a better console entry point and infrastructure for integration testing. In the terminal, you can now run `bcipy` instead of `python bci_main.py` 
- `parameters.json`: add stim_order #153 add max selections #175 remove max_inq_per_trial in favor of max_inq_per_series #176 add inquiry preview #177 with relevant stimuli units in help text, better starting stim_height, and inquiry preview keys #216
- `demo_stimuli_generation.py`: update imports and add a case showing the new ordering functionality. #153
- `copy_phrase_wrapper`: update logging and exception handling. add stim order. #153 BUGFIX: return transformed sampling rate #159
- `random_rsvp_calibration_inq_gen`: rename to `calibration_inquiry_generator` #153
- `ExperimentField.py`: updated to use new alert types with timeouts #156
- `ExperimentRegistry.py`: add the ability to toggle anonymization of field data and use new alert types with timeouts #156
- `FieldRegistry.py`: updated to use new alert types with timeout #156
- `gui/BCInterface.py`: use `load_users` method to populate user dropdown and remove internal BCInterface load method #125
- `gui/gui_main.py`: update to return a value in FormInput, set a value for IntegerInput only if provided #156
- `gui/viewer/data_viewer.py`: Replaced the original WxPython version of the signal data viewer with the new PyQt version #157. Use signal process filters instead of duplicating logic #147. 
- `gui/viewer/file_viewer.py`: to use new raw data format #160
- `bcipy/acquisition`: refactored acquisition to support multi-modal acquisition and more performant real-time acquisition. These changes were significant and across multiple PRs. Support for new raw data format #160
- `ring_buffer_test.py` -> `test_ring_buffer.py`: to comply with naming conventions #156
- `signal/model/base_model.py`: add reshaper to base model class, requiring it to be defined in all models and return a `Reshaper`. Fix return types. #147
- `signal/model/offline_analysis.py`: updated to use new reshapers and transforms. #147 updated to report execution time and logging levels #163
- `bcipy/language_model/` --> `bcipy/language/` refactor for clarity and add base class `LanguageModel` #164
- `bcipy/tasks` --> `bcipy/task`: refactor for clarity, add README, organize tasks under paradigm #162 organize tasks operation objects into `control` module #162 #178
- `task/paradigm/rsvp/copy_phrase.py`: refactored overall #146 to use new Session classes #127 and updated to use new reshapers and transforms #147 implements current state of inquiry preview #129 #177 to account for max selection parameter #175 fix targetness #179
- `bcipy/task/data.py`: to track number of decisions made #175
- `task/control/handler.py`: added the ability to set constants in defined stimuli agent #178
- `task/control/query.py`: remove redundant best_selection in favor of one with constants. Implemented constants in return_stimuli methods. #178
- `display/rsvp/display.py`: refactored to use new trigger pulse and ensure it occurs only on first display call (whether that be `do_inquiry` or `preview_inquiry`) #149 Overall refactoring of properties into `StimuliProperties`, `InformationProperties`, `TaskDisplayProperties`. Added `PreviewInquiryProperties` and `preview_inquiry` kwarg. Add full-screen parameter to help with scaling stimuli. Add textbox to self.`_create_stimulus`. Add `preview_inquiry` and `_generate_inquiry_preview` methods. #129
- `static/images/gui_images`: updated to `gui` and refactored where defined #149
- `bcipy/display/main.py`: move `StimuliProperties`, `InformationProperties`, `TaskDisplayProperties` and `PreviewInquiryProperties` to higher level #180


- `helpers/stimuli.py `: refactored for clarity and add `get_fixation` method #149 glossary updates and remove unneeded code #178 fix targetness in copy phrase #179
- `helpers/triggers.py`: refactored `_calibration_trigger for clarity` and add `CalibrationType`(deprecating sound and adding text) #149 add an offset correction method #126
- `helpers/load.py`: updated to use new raw data format #160
- `helpers/convert.py`: mode, write_targetness, and annotation_channels keyword arguments #126 add compression/decompression support for experiments and BciPy session data #173
- `helpers/session.py`: refactored session helpers to use the new Session data structures. #127
- `helpers/exceptions`: refactored Field and Experiment exceptions to corresponding base exception #156
- `feedback/auditory_feedback`: to allow for easier setting of relevant parameters and testing #128
- `feedback/visual_feedback`: deprecate shape feedback type, line_color (in the administer method), and compare assertion as both were unused and added unneeded complexity. Set hard-coded values on the class instance for easier changing later. #128

### Removed
 
- `target_rsvp_inquiry_generator`: #153 unused
- `rsvp_copy_phrase_inq_generator`: #153 unused
- `tasks/rsvp/icon_to_icon.py`: #129 unused
- `tasks/rsvp/calibration/inter_inquiry_feedback_calibration.py`: unused
- `generate_icon_match_images`: #153 deprecated task
- `signal/process/demo/text_filter_demo.py`: #147 removes old matlab generated filter
- `signal/process/filter/resources/filters.txt`: #147 in favor of new filters and transforms
- `signal/process/filter/notch.py`: #147 in favor of new filters and transforms
- `signal/process/filter/downsample.py`: #147 in favor of new filters and transforms
- `signal/process/filter/bandpass.py`: #147 in favor of new filters and transforms

# 1.5.0

## Contributions

This version contains major refactoring and tooling improvements across the codebase. In addition, it introduces the concept of BciPy Experiments and Fields. Below we describe the major changes along with a PR# in github where applicable. 

### Added
- Language model histogram #91 
- BciPy official glossary (Sequence -> Inquiry & Epoch -> Series) #121 
- System information to `system_utils` (cpu, platform, etc) #98 
- BciPy Experiments and Fields: See PRs #113 #111, #114 for more information on the additions!
- `.bcipy` system directory to support experiment and fields #100 
- support for python 3.7
- `rsvp/query_mechanisms`: to model the way we build inquiries #108 
- `Makefile`: contains useful install and development commands
- `convert`: a module for data conversions that will be useful for data sharing. Implemented a conversion function to the EDF format. #104 
- `exceptions`: a module for BciPy core exceptions

### Updated
- `acquisition`: refactored the acquisition module to separate the concept of a device (ex. DSI-24 headset) and a connection method to that device (TCP or LSL). #122 
- `setup.py`: with new repo location and CAMBI official support email 
- `offline_analysis`: to pull parameters from session file #90 
- `requirements.txt`: to the latest available #99 #107 
- `Parameters` (added help text, removed redundant parameters). Refactored to make them immutable. #101 
- `gui_main `: to use PyQt5. We will refactor all GUI code to use this in the future. After this PR, the signal viewer (WxPython) and a couple of loading functions will remain (Tk). #102 
- `BCInterface` : updated to use new gui_main methods. Added user if validations. #102  #120 
- `params_form`: moved into a parameters modules within GUI and updated to use PyQt5. #109 
- `dev_requirements`: used to be called test_requirements. It contains more than that, so we updated the name! #99 
- `README`: with relevant updates and contributors 

### Removed
- `RSVPKeyboard.py`


# 1.4.2

## Contributions

### Added

- Artifact Rejection
    - `signal.evaluate.Evaluator`
        - Evaluates sequences and flags sequences that break rules
    - `signal.evaluate.Rule`
        - Defines rules that sequences should follow
- Re-added progress indicator for offline analysis
- Scripts
    - `timing_tools.py` to help test display timing before experiments
- Tests
- Linting
- Documentation improvements

### Updated

- DAQ file writing. Instead of writing to `raw_data.csv` during task executions, optionally write `raw_data.csv` on call to `stop_acquisition`, and only write to SQLite database during task.
- `get_data` queries in `buffer_server` to increase speed on Windows machines. 
- RSVP sequence stimuli presentation. Reduces reported timing slips on Windows and Linux machines. 
    - List of stimuli to be presented is now generated before each sequence is presented, rather than generating stimuli during the sequence.
    - The screen is now only drawn once per stimulus, rather than redrawing the screen every frame.
- Signal viewer to shut down when data is no longer streaming
- `main_frame` and `copy_phrase` to fix bug that prevents copy phrase tasks from completing in fake data mode
- Target generation for icon-matching tasks has been changed to minimize instances of duplicate target stimuli over the course of an experiment
- Fixed issue with icon-matching tasks flashing to desktop on Windows machines
- `params_form.py` is now launched by creating another wxPython frame, rather than launching a subprocess

## Removed

- Module additions via `__init__`

# 1.4.1

Patch for gui.viewer module. Missing init file.

# 1.4.0

This release focused on bug fixes, exposing parameters, and refactoring. Further dual screen configuration and testing was done to allow for simultaneous signal viewing and task operation. 

## Added

    - Dual screen configuration / updated support
    - Parameters:
            - Copy phrase decision thresholds
            - Inter-sequence Feedback level thresholds
   

## Updated
    - RSVP Display: refactor 
    - Decision Maker / Evidence Fusion: refactor
    - Signal Viewer: more distinct channel names
    - bci_main: shutdown handling and bug fix
    - Language Model Helper: bug fix for negative probabilities

## Removed
    - Multicolor Text
    - Old LSL viewer


# 1.3.0

This release focused on the addition of a Signal Viewer, Inter-sequence Feedback Task, Signal Processing / Decomposition Methods, and miscellaneous cleanup. 

## Added

    - PSD: Power spectral density methods
    - DSI VR-300 integration
    - Logging session configuration and setup
    - Version and git commit extraction
    - Inter-sequence feedback task
    - Backspace frequency parameterization and implementation
    - Bandpass and Notch Filter implementation
    - Custom Exceptions
   

## Updated
    - Refactor RSVP Task: Icon-to-Icon (WIP)
    - Refactored Signal Module
    - Dependencies (PsychoPy, pylsl, numpy, pandas, WxPython)
    - Tests
    - Documentation

## Removed
    - Bar Graph implementation


# 1.2.0

This release focused on the addition of a new Alert Tone Task, integration of the Language Model, and other fixes to make Copy Phrase work better.

## New Features
    - Alert Tone Task: a new calibration task that plays an alert tone prior to displaying the presentation letter. Adds parameters for the directory in which sounds (.wav files) are located as well as the length of the delay.
    - SPACE symbol customization: allows users to customize the display of the SPACE character.
    - New Language Model: experimental; adds a new oclm Language Model and allows users to select between this and the standard prelm model. Note that this functionality is still incomplete.

## Updates
    - Language Model integration: the Language Model now correctly starts up a Docker container on both Windows and Mac machines.
    - Fixes to the CopyPhraseWrapper in its usage of the Language Model, including converting its output to the probability domain for integration, and correctly handling space characters.
    - Fix for Copy Phrase backspace selection.
    - Fixed issue loading EEG Classifier pkl files.
    - General code cleanup in the acquisition module.
    - Simplified code for starting a new task.
    - There is now a task_registry which lists all available tasks and allows code to enumerate them.
    - More obvious feedback when a CopyPhrase letter has been typed.
    - Users can now override configuration for parameters with a drop-down menu. The previous behavior was to restrict users to the suggested values.


# 1.1.0

This is a working version for use with LSL on Calibration and Copy phrase tasks. It will break implementation from previous release. Future minor versions should not do this.

## Updated:
- Structure of Signal Module
- Signal Model and Trial Reshaper Refactored
- Documentation
- Data acquisition client naming

## Added:
- Image scaling
- Initial Icon Word matching task
- Task breaks
- LSL offset correction
- Unittests
- AUC printing to filename
- logging
- Initial signal viewer
- Initial average ERP generation plots
- GUI enhancements

## Removed
- Duplicate dependencies
