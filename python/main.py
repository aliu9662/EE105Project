import serial
import time
import tkinter as tk
from tkinter import font
from collections import deque

# --- 1. Configuration & Constants ---
PORT = input("Enter the serial port (e.g., COM5 or /dev/ttyUSB0): ").strip()
THRESHOLD_PRESSURE = 100.0  # kPa
THRESHOLD_PROX_HIGH = 250
BUFFER_SIZE = 3  # Changed to 3 consecutive valid readings

# Colors
BG_NORMAL = "#0B0E14"       # Deep Obsidian
BG_WARNING = "#440000"      # Dark Blood Red
TEXT_CYAN = "#00F0FF"       # Tech Blue
TEXT_RED = "#FF003C"        # Alert Red

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
root.geometry("500x500")
root.configure(bg=BG_NORMAL)

# Global State
prox_history = deque(maxlen=BUFFER_SIZE)
is_alerting = False
flash_state = True

# Fonts
header_font = font.Font(family="Courier", size=10, weight="bold")
data_font = font.Font(family="Impact", size=32)
alert_font = font.Font(family="Courier", size=18, weight="bold")

# UI Elements
def create_label(text, color, py=10):
    lbl = tk.Label(root, text=text, font=header_font, fg=color, bg=BG_NORMAL)
    lbl.pack(pady=(10, 0))
    val = tk.Label(root, text="--", font=data_font, fg="white", bg=BG_NORMAL)
    val.pack(pady=(0, 10))
    return val

lbl_alert = tk.Label(root, text="SYSTEM STABLE", font=alert_font, fg="#00FF41", bg=BG_NORMAL)
lbl_alert.pack(pady=20)

val_pressure = create_label("ATMOSPHERIC PRESSURE (kPa)", TEXT_CYAN)
val_proximity = create_label("PROXIMITY VECTOR", "#CC00FF")
val_red = create_label("PPG RED CHANNEL", "#FF3333")
val_ir = create_label("PPG IR CHANNEL", "#33CCFF")

def toggle_flash():
    """Handles the pulsing red effect and flashing text"""
    global flash_state
    if is_alerting:
        flash_state = not flash_state
        current_bg = BG_WARNING if flash_state else "#880000"
        root.configure(bg=current_bg)
        lbl_alert.config(text="!!! WARNING: CRITICAL !!!", fg="white", bg=current_bg)
        for widget in root.winfo_children():
            widget.configure(bg=current_bg)
    else:
        root.configure(bg=BG_NORMAL)
        lbl_alert.config(text="SYSTEM STABLE", fg="#00FF41", bg=BG_NORMAL)
        for widget in root.winfo_children():
            widget.configure(bg=BG_NORMAL)
            
    root.after(400, toggle_flash)

def update_data():
    global is_alerting
    try:
        latest_line = None
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line: latest_line = line
                
        if latest_line:
            if latest_line.count(',') == 3:
                parts = latest_line.split(',')
                p_val, prox_val, r_val, i_val = map(float, parts)
                
                # Logic: Pressure Check
                pressure_low = p_val < THRESHOLD_PRESSURE
                
                # Logic: Filter Proximity and check buffer
                if prox_val != -1:
                    # Only add to history and update screen if it's a valid reading
                    prox_history.append(prox_val)
                    val_proximity.config(text=f"{int(prox_val)}")
                
                prox_danger = False
                if len(prox_history) == BUFFER_SIZE:
                    # Check if the last 3 VALID readings are all strictly greater than 250
                    if all(x > THRESHOLD_PROX_HIGH for x in prox_history):
                        prox_danger = True

                # Set Global Alert State
                is_alerting = pressure_low or prox_danger

                # Update Always-valid Displays
                val_pressure.config(text=f"{p_val:.2f}")
                val_red.config(text=f"{int(r_val)}")
                val_ir.config(text=f"{int(i_val)}")
                
            else:
                # Print non-data statuses to the console
                print(f"Status from Arduino: {latest_line}")

    except Exception as e:
        print(f"Data Stream Error: {e}")

    root.after(50, update_data)

def on_closing():
    if ser.is_open:
        ser.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.after(100, update_data)
root.after(100, toggle_flash)
root.mainloop()