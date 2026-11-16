import tkinter as tk
import subprocess
import socket
import time
import sys
import os
import json
import webbrowser

def get_python_exe():
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        candidate = exe[:-5] + ".exe"
        if os.path.exists(candidate):
            return candidate
    return exe

def is_port_in_use(port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def launch_web_dashboard():
    root.destroy()
    py = get_python_exe()
    
    # cmd /k ensures the window STAYS OPEN so you can read any errors if it crashes!
    subprocess.Popen(f'cmd.exe /k "{py}" app.py', creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    # Smart wait: checks every 0.2 seconds until the server is actually alive
    for _ in range(50):
        if is_port_in_use(5000):
            break
        time.sleep(0.2)
        
    # Opens instantly in your default browser the millisecond the server is ready
    webbrowser.open("http://127.0.0.1:5000")

def launch_meeting_overlay():
    root.destroy()
    py = get_python_exe()
    
    # Keeps window open for error reading
    subprocess.Popen(f'cmd.exe /k "{py}" meeting_overlay.py', creationflags=subprocess.CREATE_NEW_CONSOLE)

def save_config_and_launch(launch_func):
    def wrapper():
        with open("mode_config.json", "w") as f:
            json.dump({"mogging_mode": mogging_var.get()}, f)
        launch_func()
    return wrapper

root = tk.Tk()
root.title("DeepFace Behavior System — Mode Launcher")
root.configure(bg="#080d19")
root.resizable(False, False)

container = tk.Frame(root, bg="#080d19", padx=28, pady=22)
container.pack(fill="both", expand=True)

tk.Label(
    container, text="AI VISION / BEHAVIORAL ANALYSIS",
    font=("Segoe UI", 9, "bold"), fg="#8797ff", bg="#080d19"
).pack(pady=(0, 4))

tk.Label(
    container, text="Select Operating Mode",
    font=("Segoe UI", 18, "bold"), fg="#eef3ff", bg="#080d19"
).pack(pady=(0, 16))

mogging_var = tk.BooleanVar(value=False)
try:
    with open("mode_config.json", "r") as f:
        config = json.load(f)
        mogging_var.set(config.get("mogging_mode", False))
except:
    pass

chk_mog = tk.Checkbutton(
    container, 
    text="Enable 'Mogging Mode' (Geometry Censor Overlay)",
    variable=mogging_var,
    font=("Segoe UI", 10, "bold"), fg="#ff4a4a", bg="#080d19",
    selectcolor="#10182a", activebackground="#080d19", activeforeground="#ff4a4a"
)
chk_mog.pack(pady=(0, 10))

btn_web = tk.Button(
    container,
    text="1. Web Dashboard (Webcam & Image Upload)\nOpens full browser interface with live webcam & upload",
    font=("Segoe UI", 10, "bold"),
    fg="#ffffff", bg="#3847df", activebackground="#4657f3", activeforeground="#ffffff",
    relief="flat", cursor="hand2", padx=16, pady=12, width=44,
    command=save_config_and_launch(launch_web_dashboard)
)
btn_web.pack(pady=6, fill="x")

btn_overlay = tk.Button(
    container,
    text="2. Live Meeting Overlay (Always-on-Top HUD)\nSelect any call window (Zoom/Meet/Discord) & float over apps",
    font=("Segoe UI", 10, "bold"),
    fg="#080d19", bg="#42d99a", activebackground="#6ee7d1", activeforeground="#080d19",
    relief="flat", cursor="hand2", padx=16, pady=12, width=44,
    command=save_config_and_launch(launch_meeting_overlay)
)
btn_overlay.pack(pady=6, fill="x")

tk.Label(
    container, text="Both modes log CSV telemetry & plot summary graphs on exit",
    font=("Segoe UI", 8), fg="#7181a2", bg="#080d19"
).pack(pady=(12, 0))

root.update_idletasks()
w = root.winfo_reqwidth()
h = root.winfo_reqheight()
x = (root.winfo_screenwidth() // 2) - (w // 2)
y = (root.winfo_screenheight() // 2) - (h // 2)
root.geometry(f"{w}x{h}+{x}+{y}")
root.mainloop()