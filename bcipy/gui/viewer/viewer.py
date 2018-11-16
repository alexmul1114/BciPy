"""
An example of how to use wx or wxagg in an application with the new
toolbar - comment out the setA_toolbar line for no toolbar
"""
import csv
import itertools as it

import matplotlib
# uncomment the following to use wx rather than wxagg
# matplotlib.use('WX')
# from matplotlib.backends.backend_wx import FigureCanvasWx as FigureCanvas

# comment out the following to use wx rather than wxagg
matplotlib.use('WXAgg')
import numpy as np
import wx
import wx.lib.mixins.inspection as WIT
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import NullFormatter, NullLocator
from numpy import arange, pi, sin
from wx import (BOTTOM, EVT_TIMER, EXPAND, LEFT, TOP, VERTICAL, BoxSizer,
                Frame, Timer)

from bcipy.acquisition.device_info import DeviceInfo
from bcipy.gui.viewer.ring_buffer import RingBuffer


def downsample(data, factor=2):
    """Decrease the sample rate of a sequence by a given factor."""
    return np.array(data)[::factor]


class CanvasFrame(Frame):
    """GUI Frame in which data is plotted. Plots a subplot for every channel.

    Parameters:
    -----------
        data_gen : generator where each `next` value is a row of data.
        device_info: metadata about the data.
        seconds - how many seconds worth of data to display.
        downsample_factor - how much to compress the data. A factor of 1
            displays the raw data.
        refresh - time in milliseconds; how often to refresh the plots
    """

    def __init__(self, data_gen, device_info: DeviceInfo,
                 seconds: int = 2, downsample_factor: int = 2,
                 refresh: int = 500):
        Frame.__init__(self, None, -1,
                       'EEG Viewer', size=(800, 550))

        self.refresh_rate = refresh
        self.samples_per_second = device_info.fs
        self.records_per_refresh = int(
            (self.refresh_rate / 1000) * self.samples_per_second)

        self.data_gen = data_gen
        self.header = device_info.channels
        self.data_indices = [i for i in range(len(self.header))
                             if 'TRG' not in self.header[i] and 'timestamp' not in self.header[i]]

        self.seconds = seconds
        self.downsample_factor = downsample_factor
        self.buffer = self.init_buffer()

        # figure size is in inches.
        self.figure = Figure(figsize=(15, 10), dpi=80, tight_layout=True)
        self.axes = self.init_axes()

        self.canvas = FigureCanvas(self, -1, self.figure)
        self.timer = Timer(self)
        self.Bind(EVT_TIMER, self.update_data, self.timer)

        self.CreateStatusBar()

        # Toolbar
        self.toolbar = wx.BoxSizer(wx.HORIZONTAL)
        self.start_stop_btn = wx.Button(self, -1, "Start")
        self.downsample_checkbox = wx.CheckBox(self, label="Downsampled")
        self.downsample_checkbox.SetValue(downsample_factor > 1)

        self.Bind(wx.EVT_BUTTON, self.toggle_stream, self.start_stop_btn)
        self.Bind(wx.EVT_CHECKBOX, self.toggle_downsampling,
                  self.downsample_checkbox)
        self.toolbar.Add(self.start_stop_btn, 1, wx.ALIGN_CENTER, 0)
        self.toolbar.Add(self.downsample_checkbox, 1, wx.ALIGN_CENTER, 0)

        self.sizer = BoxSizer(VERTICAL)
        self.sizer.Add(self.canvas, 1, LEFT | TOP | EXPAND)
        self.sizer.Add(self.toolbar, 0, wx.ALIGN_BOTTOM | wx. ALIGN_CENTER)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.Fit()
        self.init_data()

        self.started = True
        self.start()

    def init_buffer(self):
        buf_size = int((self.samples_per_second * self.seconds) /
                       self.downsample_factor)
        return RingBuffer(buf_size, pre_allocated=True)

    def init_axes(self):
        axes = self.figure.subplots(len(self.data_indices), 1, sharex=True)
        """Sets configuration for axes"""
        for i, ch in enumerate(self.data_indices):
            ch_name = self.header[ch]
            axes[i].set_frame_on(False)
            axes[i].set_ylabel(ch_name, rotation=0, labelpad=15)
            # if i == len(self.data_indices) - 1:
            #     self.axes[i].set_xlabel("Sample")
            axes[i].yaxis.set_major_locator(NullLocator())
            axes[i].xaxis.set_major_formatter(NullFormatter())
            # x-axis label
            axes[i].yaxis.set_major_formatter(NullFormatter())
            # self.axes[i].xaxis.set_major_locator(NullLocator())
            axes[i].grid()
        return axes

    def reset_axes(self):
        self.figure.clear()
        self.axes = self.init_axes()

    def start(self):
        """Start streaming data in the viewer."""
        self.timer.Start(self.refresh_rate)
        self.started = True
        self.start_stop_btn.SetLabel("Pause")

    def stop(self):
        """Stop/Pause the viewer."""
        self.timer.Stop()
        self.started = False
        self.start_stop_btn.SetLabel("Start")

    def toggle_stream(self, event):
        """Toggle data streaming"""
        if self.started:
            self.stop()
        else:
            self.start()

    def toggle_downsampling(self, event):
        """Toggle whether or not the data gets downsampled"""
        # TODO: use original configured downsample_factor
        if self.downsample_checkbox.GetValue():
            self.downsample_factor = 2
        else:
            self.downsample_factor = 1
        previously_running = self.started
        if self.started:
            self.stop()
        # re-initialize
        self.buffer = self.init_buffer()
        self.reset_axes()
        self.init_data()
        if previously_running:
            self.start()

    def update_buffer(self):
        """Update the buffer with latest data and return the data"""
        try:
            records = list(it.islice(self.data_gen, self.records_per_refresh))
            for row in downsample(records, self.downsample_factor):
                self.buffer.append(row)
        except StopIteration:
            self.stop()
        return self.buffer.get()

    def data_for_channel(self, ch, rows):
        """Extract the data for a given channel"""
        return [0.0 if r is None else float(r[ch]) for r in rows]

    def init_data(self):
        """Initialize the data."""
        rows = self.update_buffer()

        # plot each channel
        for i, ch in enumerate(self.data_indices):
            data = self.data_for_channel(ch, rows)
            self.axes[i].plot(data, linewidth=0.8)

    def update_data(self, evt):
        """Called by the timer on refresh."""
        rows = self.update_buffer()

        # TODO: more efficient method of splitting out channels
        # plot each channel
        for i, ch in enumerate(self.data_indices):
            data = self.data_for_channel(ch, rows)
            self.axes[i].lines[0].set_ydata(data)
            self.axes[i].set_ybound(lower=min(data), upper=max(data))

        self.canvas.draw()


def main(data_file: str, seconds: int, downsample: int, refresh: int):
    """Run the viewer gui

    Parameters:
    -----------
        data_file - raw_data.csv file to stream.
        seconds - how many seconds worth of data to display.
        downsample - how much the data is downsampled. A factor of 1
            displays the raw data.
    """
    with open(data_file) as csvfile:
        # read metadata
        name = next(csvfile).strip().split(",")[-1]
        fs = float(next(csvfile).strip().split(",")[-1])

        reader = csv.reader(csvfile)
        channels = next(reader)

        app = wx.App(False)
        frame = CanvasFrame(reader, DeviceInfo(
            fs=fs, channels=channels, name=name), seconds, downsample,
            refresh)
        frame.Show(True)
        app.MainLoop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', help='path to the data file', default='raw_data.csv')
    parser.add_argument('-s', '--seconds',
                        help='seconds to display', default=2, type=int)
    parser.add_argument('-d', '--downsample',
                        help='downsample factor', default=2, type=int)
    parser.add_argument('-r', '--refresh',
                        help='refresh rate in ms', default=500, type=int)
    args = parser.parse_args()
    main(args.file, args.seconds, args.downsample, args.refresh)
