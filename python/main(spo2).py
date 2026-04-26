import serial
import time
import tkinter as tk
from tkinter import font
from collections import deque
import numpy as np


# --- 1. Configuration & Constants ---
PORT = input("/dev/cu.usbmodem1101").strip()
THRESHOLD_PRESSURE = 100.0 
THRESHOLD_PROX_HIGH = 250
BUFFER_SIZE = 3
SPO2_WINDOW_SIZE = 100 # Samples per calculation (approx 5 seconds at 20Hz)

# Colors
BG_NORMAL = "#0B0E14"
BG_WARNING = "#440000"
TEXT_CYAN = "#00F0FF"
TEXT_GOLD = "#FFD700"
TEXT_GREEN = "#00FF41" # Healthy
TEXT_ORANGE = "#FF8C00" # Warning
TEXT_RED = "#FF003C"    # Danger

try:
    ser = serial.Serial(PORT, 115200, timeout=0.05) 
    time.sleep(2) 
except Exception as e:
    print(f"Failed to connect. Error: {e}")
    exit()

print("="*50)
print("SYSTEM INITIALIZING... ACTIVATE PPG SENSOR ('m3ppg')")
print("="*50)
command = input("Type command (>): ").strip()
ser.write((command + '\n').encode())

# --- 2. UI Setup ---
root = tk.Tk()
root.title("ADVANCED SENSOR TELEMETRY")
root.geometry("500x750") 
root.configure(bg=BG_NORMAL)

# Global State
prox_history = deque(maxlen=BUFFER_SIZE)
red_buffer = []
ir_buffer = []
is_alerting = False
flash_state = True

# Fonts
header_font = font.Font(family="Courier", size=10, weight="bold")
data_font = font.Font(family="Impact", size=32)
status_font = font.Font(family="Courier", size=14, weight="bold")
alert_font = font.Font(family="Courier", size=18, weight="bold")

# UI Elements
def create_label(text, color):
    lbl = tk.Label(root, text=text, font=header_font, fg=color, bg=BG_NORMAL)
    lbl.pack(pady=(10, 0))
    val = tk.Label(root, text="--", font=data_font, fg="white", bg=BG_NORMAL)
    val.pack(pady=(0, 5))
    return val

lbl_alert = tk.Label(root, text="SYSTEM STABLE", font=alert_font, fg=TEXT_GREEN, bg=BG_NORMAL)
lbl_alert.pack(pady=20)

val_pressure = create_label("ATMOSPHERIC PRESSURE (kPa)", TEXT_CYAN)
val_proximity = create_label("PROXIMITY VECTOR", "#CC00FF")

# SpO2 Section
val_spo2 = create_label("CALCULATED SpO2 (%)", TEXT_GOLD)
lbl_spo2_status = tk.Label(root, text="WAITING FOR DATA...", font=status_font, fg="gray", bg=BG_NORMAL)
lbl_spo2_status.pack(pady=(0, 10))

val_red = create_label("PPG RED CHANNEL", "#FF3333")
val_ir = create_label("PPG IR CHANNEL", "#33CCFF")

def get_spo2_status(val):
    """Returns the text status and color for a given SpO2 value."""
    if isinstance(val, str): return "---", "gray"
    if val >= 95:
        return "STATUS: NORMAL", TEXT_GREEN
    elif 90 <= val < 95:
        return "STATUS: LOW / CAUTION", TEXT_ORANGE
    else:
        return "STATUS: CRITICAL", TEXT_RED

def calculate_spo2(r_samples, i_samples):
    if len(r_samples) < SPO2_WINDOW_SIZE:
        return "--"
    try:
        dc_red = np.mean(r_samples)
        dc_ir = np.mean(i_samples)
        ac_red = np.max(r_samples) - np.min(r_samples)
        ac_ir = np.max(i_samples) - np.min(i_samples)
        
        if dc_red == 0 or dc_ir == 0 or ac_ir == 0: return "ERR"

        R = (ac_red / dc_red) / (ac_ir / dc_ir)
        spo2_val = 110 - 25 * R # Empirical linear formula
        return max(1, min(100, int(spo2_val)))
    except:
        return "ERR"

def toggle_flash():
    global flash_state
    if is_alerting:
        flash_state = not flash_state
        current_bg = BG_WARNING if flash_state else "#880000"
        root.configure(bg=current_bg)
        lbl_alert.config(text="!!! WARNING: CRITICAL !!!", fg="white", bg=current_bg)
    else:
        root.configure(bg=BG_NORMAL)
        lbl_alert.config(text="SYSTEM STABLE", fg=TEXT_GREEN, bg=BG_NORMAL)
    
    # Refresh all widgets to match the flashing background
    current_color = root.cget("bg")
    for widget in root.winfo_children():
        widget.configure(bg=current_color)
            
    root.after(400, toggle_flash)

def update_data():
    global is_alerting, red_buffer, ir_buffer
    try:
        latest_line = None
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line: latest_line = line
                
        if latest_line and latest_line.count(',') == 3:
            parts = latest_line.split(',')
            p_val, prox_val, r_val, i_val = map(float, parts)
            
            red_buffer.append(r_val)
            ir_buffer.append(i_val)
            
            if len(red_buffer) >= SPO2_WINDOW_SIZE:
                result = calculate_spo2(red_buffer, ir_buffer)
                val_spo2.config(text=f"{result}")
                
                # Update SpO2 Health Status Text
                status_text, status_color = get_spo2_status(result)
                lbl_spo2_status.config(text=status_text, fg=status_color)
                
                red_buffer = []
                ir_buffer = []

            # Alert Logic
            pressure_low = p_val < THRESHOLD_PRESSURE
            prox_danger = False
            if prox_val != -1:
                prox_history.append(prox_val)
                val_proximity.config(text=f"{int(prox_val)}")
                if len(prox_history) == BUFFER_SIZE and all(x > THRESHOLD_PROX_HIGH for x in prox_history):
                    prox_danger = True

            is_alerting = pressure_low or prox_danger

            # Update Labels
            val_pressure.config(text=f"{p_val:.2f}")
            val_red.config(text=f"{int(r_val)}")
            val_ir.config(text=f"{int(i_val)}")

    except Exception as e:
        print(f"Data Stream Error: {e}")

    root.after(50, update_data)

def on_closing():
    if ser.is_open: ser.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.after(100, update_data)
root.after(100, toggle_flash)
root.mainloop()