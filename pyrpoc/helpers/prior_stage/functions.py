from ctypes import WinDLL, create_string_buffer
import os
import time
import numpy as np
from pyrpoc.mains import acquisition
import cv2

DLL_PATH = os.path.join(os.path.dirname(__file__), "PriorScientificSDK.dll")
SDKPrior = None
sessionID = None 
_prior_connected = False  # Track connection state


def connect_prior(port=4):
    global _prior_connected
    if _prior_connected:
        return

    ret, _ = send_command(f"controller.connect {port}")
    if ret == 0:
        _prior_connected = True
        print(f"Connected to Prior stage on COM{port}")
    else:
        raise RuntimeError(f"Failed to connect to Prior stage on COM{port}")


def initialize_sdk():
    global SDKPrior, sessionID

    if SDKPrior is None:
        if os.path.exists(DLL_PATH):
            SDKPrior = WinDLL(DLL_PATH)
        else:
            raise RuntimeError("DLL could not be loaded.")

        ret = SDKPrior.PriorScientificSDK_Initialise()  
        if ret != 0:
            raise RuntimeError(f"Failed to initialize Prior SDK. Error code: {ret}")

        print("Prior SDK Initialized.")

    if sessionID is None:
        sessionID = SDKPrior.PriorScientificSDK_OpenNewSession()
        if sessionID < 0:
            raise RuntimeError(f"Failed to open Prior SDK session. SessionID: {sessionID}")

        print(f"SDK Session Opened. Session ID: {sessionID}")


def send_command(command):
    initialize_sdk()

    rx = create_string_buffer(1000)
    ret = SDKPrior.PriorScientificSDK_cmd(
        sessionID, create_string_buffer(command.encode()), rx
    )
    response = rx.value.decode().strip()

    if ret != 0:
        print(f"Error executing command: {command} (Return Code: {ret})")

    return ret, response


def wait_for_z_motion():
    while True:
        _, response = send_command("controller.z.busy.get")

        if response:
            try:
                status = int(response)
                if status == 0:
                    break  
            except ValueError:
                print(f"Invalid response from controller: '{response}'")
        else:
            print("No response from controller, is it connected?")

        time.sleep(0.1)

def auto_focus(gui, port: int, channel_name: str, step_size=10):
    connect_prior(port)

    gui.simulation_mode.set(False)
    gui.acquiring = True

    try:
        channel_index = gui.config["channel_names"].index(channel_name)
    except ValueError:
        raise RuntimeError(f"Invalid channel name: '{channel_name}'")

    ret, current_z = send_command("controller.z.position.get")
    if ret != 0:
        raise RuntimeError("Failed to retrieve current Z position.")
    try:
        current_z = int(current_z)
    except ValueError:
        raise RuntimeError(f"Invalid Z position response: '{current_z}'")

    z_positions = [current_z + i * step_size for i in range(-10, 11)]  # total of 21 points
    best_focus = -1
    best_z = current_z

    gui.progress_label.config(text=f'(0/{len(z_positions)})')
    gui.root.update_idletasks()

    for i, z in enumerate(z_positions):
        if not gui.acquiring:
            print("[Autofocus] Interrupted by Stop.")
            break

        try:
            send_command(f"controller.z.goto-position {z}")
            wait_for_z_motion()
        except Exception as e:
            gui.acquiring = False
            raise RuntimeError(f"Stage move to Z={z} failed: {e}")

        acquisition.acquire(gui, auxilary=True)
        gui.root.update_idletasks()
        gui.root.update()

        image = gui.data[channel_index]
        x, y = np.shape(image)
        metric = cv2.Laplacian(image[int(3*x/8):int(5*x/8), int(3*y/8):int(5*y/8)], cv2.CV_64F).var() # only use the middle eighth for focus
        print(f"Z={z} → Focus Metric={metric:.2f}")

        if metric > best_focus:
            best_focus = metric
            best_z = z

        gui.progress_label.config(text=f'({i + 1}/{len(z_positions)})')
        gui.root.update_idletasks()

    if gui.acquiring:
        send_command(f"controller.z.goto-position {best_z}")
        wait_for_z_motion()

        acquisition.acquire(gui, auxilary=True)
        gui.root.update_idletasks()
        gui.root.update()
        print(f"[Autofocus] Best Z = {best_z}, Metric = {best_focus:.2f}")

    gui.acquiring = False
    return best_z, best_focus


def move_z(port: int, z_height: int):
    connect_prior(port)

    if not (0 <= z_height <= 50000):
        raise ValueError("Z height must be between 0 and 50,000 µm.")

    ret, _ = send_command(f"controller.z.goto-position {z_height}")
    if ret != 0:
        raise RuntimeError(f"Could not move Prior stage to {z_height} µm.")
    wait_for_z_motion()


def move_xy(port: int, x: int, y: int):
    connect_prior(port)

    if not (0 <= x <= 50000) or not (0 <= y <= 50000):
        raise ValueError("X and Y positions must be between 0 and 50,000 µm.")

    ret, _ = send_command(f"controller.stage.goto-position {x} {y}")
    if ret != 0:
        raise RuntimeError(f"Could not move Prior stage to {x}, {y}.")


if __name__ == "__main__":
    print("connecting")
    connect_prior(4)

    move_xy(4, 1000, 1000)
    move_z(4, 2500)
