from psychopy import visual, core

class VEP(object):    
    def __init__(self, win,
                codes=[0,1],
                frame_on=1,
                frame_off=1,
                trialdur = 5.0,
                inter_trial_period=1,
                position=[0, 0],
                colors = ('white', 'black')):
        
        self.win = win
        position = position
        size = 5
        self.pattern1 = visual.GratingStim(win=self.win, name='pattern1',units='cm', 
                        tex=None, mask=None,
                        ori=0, pos=position, size=size, sf=1, phase=0.0,
                        color=colors[0], colorSpace='rgb', opacity=1, 
                        texRes=256, interpolate=True, depth=-1.0)
        self.pattern2 = visual.GratingStim(win=self.win, name='pattern2',units='cm', 
                        tex=None, mask=None,
                        ori=0, pos=position, size=size, sf=1, phase=0,
                        color=colors[1], colorSpace='rgb', opacity=1,
                        texRes=256, interpolate=True, depth=-2.0)
        self.fixation = visual.TextStim(win=self.win, height=5, pos=[0,0], text='+', color='red')
        self.frame_on = frame_on
        self.frame_off = frame_off
        self.trialdur = trialdur
        self.inter_trial_period = inter_trial_period
        self.fixation_period = 2
        self.codes = codes

    def start_code_vep(self):

        self.framerate = self.win.getActualFrameRate()
        print(f'{self.win.getActualFrameRate()}')
        self.Trialclock = core.Clock()

        self.fixation.draw()
        self.win.flip()
        core.wait(self.fixation_period)
        while self.Trialclock.getTime() < self.trialdur:
            for code in self.codes:
                if code == 0:
                    self.frames_on()
                else:
                    self.frames_off()

            
        self.win.flip()
        core.wait(self.inter_trial_period)
        self.Trialclock.reset()    

    def frames_on(self):
        for _ in range(self.frame_on):
            self.pattern1.draw()
            self.win.flip()
    
    def frames_off(self):        
        for _ in range(self.frame_off):
            self.pattern2.draw()
            self.win.flip()


# things needed from user for each VEP to support multiple: position, colors, style (single stim or a checkerboard), codes;
#  then, we could loop over the registered VEPS and check each code to see if it should be drawn.

if __name__ == '__main__':

    
    win = visual.Window([800, 600], fullscr=False, monitor='testMonitor',units='deg')
    codes = [0] * 10 + [1] * 20 + [0] * 30 + [1] + [0]
    codeVEP = VEP(win, codes=codes)

    codeVEP.start_code_vep()

    win.close()
    core.quit()