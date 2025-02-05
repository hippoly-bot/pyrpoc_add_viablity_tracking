from ctypes import WinDLL, create_string_buffer
import os
import time

DLL_PATH = r"C:\Users\Lab Admin\Documents\PythonStuff\pysrs\pysrs\instruments\prior_stage\PriorScientificSDK.dll"
SDKPrior = None
sessionID = None 

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
    initialize_sdk()  # error code -10200 heheheheheehruiaewrgilaeuwblaiewjghlkajgbla,knja,ekjb

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

if __name__ == "__main__":
    print("connecting")
    send_command("controller.connect 4")

    send_command(f"controller.z.goto-position 10000") 
    wait_for_z_motion()  
    _, current_pos = send_command("controller.z.position.get") 
    print(f"z pos after move: {current_pos}")

    print("disconnectiong")
    send_command("controller.disconnect")
