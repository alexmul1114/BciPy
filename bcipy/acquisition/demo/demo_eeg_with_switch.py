"""Sample script to demonstrate usage of LSL client and server."""
import subprocess
import time

from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.datastream.tcp_server import await_start
from bcipy.acquisition.datastream.mock.switch import switch_device
from bcipy.acquisition.devices import preconfigured_device
from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
from bcipy.helpers.system_utils import log_to_stdout


def start_switch():
    """Start the demo switch"""
    return subprocess.Popen('python bcipy/acquisition/datastream/mock/switch.py',
                            shell=True)


def main(debug: bool = False):
    # pylint: disable=too-many-locals
    """Creates a sample lsl client that reads data from a sample TCP server
    (see demo/server.py).

    The client/server can be stopped with a Keyboard Interrupt (Ctl-C)."""

    if debug:
        log_to_stdout()

    eeg_device = preconfigured_device('DSI-24')
    eeg_server = LslDataServer(device_spec=eeg_device)
    eeg_client = LslAcquisitionClient(device_spec=eeg_device,
                                      save_directory='.')
    switch_client = LslAcquisitionClient(device_spec=switch_device(),
                                         save_directory='.')
    await_start(eeg_server)

    # Open the Demo Switch GUI.
    start_switch()
    # Wait for switch start
    print("Waiting for Switch")
    time.sleep(0.5)
    try:
        seconds = 5
        switch_client.start_acquisition()
        eeg_client.start_acquisition()
        print(f"\nCollecting data for {seconds}s...",
              "Click in the demo switch GUI to register a switch hit.",
              "Close the GUI when finished.\n")

        time.sleep(seconds)

        eeg_client.stop_acquisition()
        switch_client.stop_acquisition()
        eeg_server.stop()
        print("\nThe collected data has been written to the local directory")

    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        eeg_client.stop_acquisition()
        switch_client.stop_acquisition()
        eeg_server.stop()
        print("\nThe collected data has been written to the local directory")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args.debug)
