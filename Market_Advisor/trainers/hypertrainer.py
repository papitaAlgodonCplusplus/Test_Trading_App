import subprocess
import platform
import time
import signal
import os
import sys

def run_script(command):
    system_platform = platform.system()

    if system_platform == "Windows":
        return subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    elif system_platform in ["Darwin", "Linux"]:
        return subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    else:
        raise Exception("Unsupported Operating System")

def terminate_process(process):
    try:
        if process.poll() is None:  # Check if the process is still running
            if platform.system() == "Windows":
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            print(f"Terminated process with PID {process.pid}")
        else:
            print(f"Process with PID {process.pid} has already exited.")
    except Exception as e:
        print(f"Failed to terminate process: {e}")

def signal_handler(sig, frame):
    print("\nCtrl + C detected. Terminating all scripts...")
    terminate_process(process1)
    terminate_process(process2)
    terminate_process(process3)
    terminate_process(process4)
    print("All scripts have been terminated.")
    sys.exit(0)

if __name__ == "__main__":
    # Path to the scripts
    real_time_script = ["python", "real_time_data_simulator.py"]
    main_script = ["python", "main.py"]
    main2_script = ["python", "main2.py"]
    main3_script = ["python", "main3.py"]

    # Register the signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("Starting main.py...")
        process2 = run_script(main_script)
        
        print("Starting main2.py...")
        process3 = run_script(main2_script)
        
        print("Starting main3.py...")
        process4 = run_script(main3_script)
        
        time.sleep(5)
        
        print("Starting real_time_data_simulator.py...")
        process1 = run_script(real_time_script)

        # Run for 1200 minutes
        timeout = 1200 * 60
        print(f"All scripts will run for {timeout / 60} minutes...")
        time.sleep(timeout)

        print("Time's up! Terminating all scripts...")
        terminate_process(process1)
        terminate_process(process2)
        terminate_process(process3)
        terminate_process(process4)

        print("All scripts have been terminated.")

    except Exception as e:
        print(f"Error occurred: {e}")
