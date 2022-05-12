import logging
from typing import List
from bcipy.display import (
    BCIPY_LOGO_PATH,
    Display,
    InformationProperties,
    StimuliProperties,
    TaskDisplayProperties,
)
from psychopy import visual
from bcipy.helpers.clock import Clock
from bcipy.helpers.task import SPACE_CHAR, get_key_press
from bcipy.helpers.triggers import TriggerCallback, _calibration_trigger



class VEPDisplay(Display):

    def __init__(
            self,
            window: visual.Window,
            static_clock,
            experiment_clock: Clock,
            stimuli: StimuliProperties,
            task_display: TaskDisplayProperties,
            info: InformationProperties,
            trigger_type: str = 'text',
            space_char: str = SPACE_CHAR,
            full_screen: bool = False):
        self.window = window
        self.window_size = self.window.size  # [w, h]
        self.refresh_rate = window.getActualFrameRate()

        # NOTE flicker rate determined by the frame rate of a monitor
        # if we define a predetermind code formation we can map that to frame number.

        # IF SSVEP, we determine frame number.. warn about limitations in log? Ex. can't do 2.4 frames

        self.logger = logging.getLogger(__name__)

        # Stimuli parameters, these are set on display in order to allow
        #  easy updating after defintion
        self.stimuli_inquiry = stimuli.stim_inquiry
        self.stimuli_colors = stimuli.stim_colors
        self.stimuli_timing = stimuli.stim_timing
        self.stimuli_font = stimuli.stim_font
        self.stimuli_height = stimuli.stim_height
        self.stimuli_pos = stimuli.stim_pos
        self.is_txt_stim = stimuli.is_txt_stim
        self.stim_length = stimuli.stim_length

        self.full_screen = full_screen

        self.staticPeriod = static_clock

        # Trigger handling
        self.first_run = True
        self.first_stim_time = None
        self.trigger_type = trigger_type
        self.trigger_callback = TriggerCallback()
        self.experiment_clock = experiment_clock

        # Callback used on presentation of first stimulus.
        self.first_stim_callback = lambda _sti: None
        self.size_list_sti = []  # TODO force initial size definition
        self.space_char = space_char  # TODO remove and force task to define
        self.task_display = task_display
        self.task = task_display.build_task(self.window)

        # Create multiple text objects based on input
        self.info = info
        self.info_text = info.build_info_text(window)

        # Create initial stimuli object for updating
        self.sti = stimuli.build_init_stimuli(window)

        self.ssvep_windows = 1 # we want several configurations


    def do_inquiry(self) -> List[float]:
        pass
    
    def wait_screen(self) -> None:
        pass

    def update_task(self) -> None:
        pass

    def flicker(self) -> None:
        inquiry_duration = 10 
