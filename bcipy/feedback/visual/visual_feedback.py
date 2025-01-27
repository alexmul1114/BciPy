from enum import Enum

from psychopy import core, visual

from bcipy.feedback.feedback import Feedback
from bcipy.helpers.clock import Clock
from bcipy.helpers.stimuli import resize_image


class FeedbackType(Enum):
    TEXT = 'TEXT'
    IMAGE = 'IMAGE'


class VisualFeedback(Feedback):
    """Visual Feedback."""

    def __init__(self, display, parameters, clock):

        # Register Feedback Type
        self.feedback_type = 'Visual Feedback'

        super(VisualFeedback, self).__init__(self.feedback_type)

        # Display Window
        self.display = display

        # Parameters
        self.parameters = parameters
        self.font_stim = self.parameters['feedback_font']
        self.height_stim = self.parameters['feedback_stim_height']

        pos_x = self.parameters['feedback_pos_x']
        pos_y = self.parameters['feedback_pos_y']

        self.pos_stim = (pos_x, pos_y)

        self.feedback_length = self.parameters['feedback_flash_time']
        self.color = self.parameters['feedback_color']

        # Clock
        self.clock = clock

        self.feedback_timestamp_label = 'visual_feedback'

    def administer(
            self,
            stimulus,
            stimuli_type=FeedbackType.TEXT):
        """Administer.

        Administer visual feedback. Timing information from parameters,
            current feedback given by stimulus.
        """

        stim = self._construct_stimulus(
            stimulus,
            self.pos_stim,
            self.color,
            stimuli_type)

        time = self._show_stimuli(stim)

        core.wait(self.feedback_length)

        return [time]

    def _show_stimuli(self, stimulus) -> None:
        stimulus.draw()
        time = [self.feedback_timestamp_label, self.clock.getTime()]  # TODO: use callback for better precision
        self.display.flip()
        return time

    def _resize_image(self, stimuli, display_size, stimuli_height):
        return resize_image(
            stimulus, display_size, stimuli_height)

    def _construct_stimulus(self, stimulus, pos, fill_color, stimuli_type):
        if stimuli_type == FeedbackType.IMAGE:
            image_stim = visual.ImageStim(
                win=self.display,
                image=stimulus,
                mask=None,
                pos=pos,
                ori=0.0)
            image_stim.size = self._resize_image(
                stimulus, self.display.size, self.height_stim)
            return image_stim
        if stimuli_type == FeedbackType.TEXT:
            return visual.TextStim(
                win=self.display,
                font=self.font_stim,
                text=stimulus,
                height=self.height_stim,
                pos=pos,
                color=fill_color)


if __name__ == "__main__":
    import argparse
    from bcipy.helpers.load import load_json_parameters
    from bcipy.display import init_display_window

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters',
                        default='bcipy/parameters/parameters.json',
                        help='Parameter location. Must be in parameters directory. \
                          Pass as bcipy/parameters/parameters.json')

    args = parser.parse_args()

    # Load a parameters file
    parameters = load_json_parameters(args.parameters, value_cast=True)
    display = init_display_window(parameters)
    clock = Clock()
    # Start Visual Feedback
    visual_feedback = VisualFeedback(
        display=display, parameters=parameters, clock=clock)
    stimulus = 'Test'
    timing = visual_feedback.administer(
        stimulus, stimuli_type=FeedbackType.TEXT)
    print(timing)
    print(visual_feedback._type())
