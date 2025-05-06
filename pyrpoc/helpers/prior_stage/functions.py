from ctypes import WinDLL, create_string_buffer
import os
import time
import numpy as np
from pyrpoc.mains import acquisition
import cv2
from skimage.measure import shannon_entropy

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

def auto_focus(gui, port: int, channel_name: str, step_size=10, numsteps=21, method='tenengrad'):
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

    z_positions = [current_z + i * step_size for i in range(0 - int(numsteps/2), 0 + int(numsteps/2))]  # total of 21 points
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

        print(np.mean(gui.data))
        image = gui.data[channel_index]
        print(np.mean(image))
        x, y = np.shape(image)

        gray = (image * 255).clip(0, 255).astype(np.uint8)
    
        if method == 'tenengrad':
            print("Image min/max:", gray.min(), gray.max(), "dtype:", gray.dtype)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            metric = np.mean(np.sqrt(sobelx**2 + sobely**2))
        elif method == 'brenner':
            shifted = np.roll(gray, -2, axis=0)
            metric = np.sum((gray - shifted)**2)
        elif method == 'entropy':
            metric = shannon_entropy(gray)
        elif method == 'sobel_variance':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            metric = np.mean(sobel_magnitude) + np.var(gray)
        elif method == 'local_variance':
            mean = cv2.blur(gray.astype(np.float32), (5, 5))
            squared_mean = cv2.blur((gray.astype(np.float32))**2, (5, 5))
            variance = squared_mean - mean**2
            metric = np.mean(variance)
        else:
            metric = cv2.Laplacian(gray, cv2.CV_64F).var() 
         
        print(f"Z={z} → {method}={metric}")

        if metric > best_focus:
            best_focus = metric
            best_z = z

        gui.progress_label.config(text=f'({i + 1}/{len(z_positions)})')
        gui.root.update_idletasks()

    if gui.acquiring:
        send_command(f"controller.z.goto-position {best_z}")
        wait_for_z_motion()
        print(f'moved to {best_z}')

        acquisition.acquire(gui, auxilary=True)
        gui.root.update_idletasks()
        gui.root.update()
        print(f"[Autofocus] Best Z = {best_z}, Metric = {best_focus:.2f}")

    gui.acquiring = False
    return best_z, best_focus


def estimate_fov(gui, port: int, channel_name: str, step_um: int = 5, iterations: int = 5) -> float:
    connect_prior(port)
    gui.simulation_mode.set(False)
    gui.acquiring = True

    channel_index = gui.config["channel_names"].index(channel_name)
    x0, y0 = get_xy(port)

    pixel_shifts = []

    for i in range(iterations):
        x_start = x0 + i * 2 * step_um  # increase distance each time to avoid reusing same region
        move_xy(port, x_start, y0)
        time.sleep(0.2)
        acquisition.acquire(gui, auxilary=True)
        ref_img = gui.data[channel_index].astype(np.float32)

        move_xy(port, x_start + step_um, y0)
        time.sleep(0.2)
        acquisition.acquire(gui, auxilary=True)
        shifted_img = gui.data[channel_index].astype(np.float32)

        shift, _ = cv2.phaseCorrelate(ref_img, shifted_img)
        delta_pixels = abs(shift[0])

        if delta_pixels == 0:
            raise RuntimeError(f"Phase correlation failed on iteration {i+1} — no shift detected")
        pixel_shifts.append(delta_pixels)

    gui.acquiring = False

    median_pixel_shift = np.median(pixel_shifts)
    pixel_size_um = step_um / median_pixel_shift
    fov_um = ref_img.shape[1] * pixel_size_um
    return fov_um



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

    x0, y0 = get_xy(port)
    if not (x0-10000 <= x <= x0+10000) or not (y0-10000 <= y <= y0+10000):
        raise ValueError("Entered position is more than 1 cm away, and may be unsafe. Cancelling...")

    ret, _ = send_command(f"controller.stage.goto-position {x} {y}")
    if ret != 0:
        raise RuntimeError(f"Could not move Prior stage to {x}, {y}.")

def get_xy(port: int):
    # returns x, y as integers
    # e.g., x0, y0 = get_xy(4)
    connect_prior(port)
    ret, response = send_command("controller.stage.position.get")
    if ret != 0:
        raise RuntimeError("Failed to get Y position.")
    try:
        return map(int, response.split(","))
    except ValueError:
        raise RuntimeError(f"Invalid Y position response: '{response}'")

def get_z(port: int):
    connect_prior(port)
    ret, response = send_command("controller.z.position.get")
    if ret != 0:
        raise RuntimeError("Failed to get Z position.")
    try:
        return int(response)
    except ValueError:
        raise RuntimeError(f"Invalid Z position response: '{response}'")

if __name__ == "__main__":
    print("connecting")
    connect_prior(4)

    print("Current Position:")
    x0, y0 = get_xy(4)
    print(x0, y0)