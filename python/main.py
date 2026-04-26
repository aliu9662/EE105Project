import serial
import time
import tkinter as tk
from tkinter import font
from collections import deque
import numpy as np
from scipy.fft import fft, fftfreq

# --- 1. Configuration & Constants ---
PORT = input("Enter the serial port (e.g., COM5 or /dev/ttyUSB0): ").strip()
THRESHOLD_PRESSURE = 100.0  # kPa
THRESHOLD_PROX_HIGH = 250
BUFFER_SIZE = 3  # Changed to 3 consecutive valid readings

# PPG Processing Constants
SAMPLE_RATE = 100       # Adjust this to match the Hz of your Arduino output
PPG_BUFFER_SIZE = 200   # Stores the last ~2 seconds of data for FFT analysis
MIN_SIGNAL_THRESH = 2000

# Colors
BG_NORMAL = "#0B0E14"       # Deep Obsidian
BG_WARNING = "#440000"      # Dark Blood Red
TEXT_CYAN = "#00F0FF"       # Tech Blue
TEXT_RED = "#FF003C"        # Alert Red
TEXT_YELLOW = "#FFD700"     # SpO2
TEXT_PINK = "#FF69B4"       # Heart Rate

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

# --- 2. Medical Calculation Logic ---
def calculate_ros(red_signal, ir_signal):
    """
    Calculates the Ratio of Ratios (R) according to the EE105 lab manual.
    """
    # DC Component: The steady, constant part of the signal (the mean)
    dc_red = np.mean(red_signal)
    dc_ir = np.mean(ir_signal)
    
    # Prevent division by zero
    if dc_red == 0 or dc_ir == 0:
        return 0
        
    # AC Component: The amplitude of the pulsating part of the signal
    # We use peak-to-peak (max - min) to find the full amplitude
    ac_red = np.max(red_signal) - np.min(red_signal)
    ac_ir = np.max(ir_signal) - np.min(ir_signal)
    
    if ac_ir == 0:
        return 0
        
    # Ratio of Ratios (R) formula
    r_val = (ac_red / dc_red) / (ac_ir / dc_ir)
    return r_val

def calculate_spo2(r_val):
    """
    Calculates SpO2 using the simplified calibration curve formula.
    """
    if r_val == 0:
        return 0
        
    # SpO2 Calibration Curve Formula: SpO2(%) = 110 - 25 * R
    spo2 = 110 - (25 * r_val)
    
    # Cap the value at a realistic maximum of 100%
    return min(spo2, 100.0)


def calculate_heart_rate(signal, sample_rate):
    n = len(signal)
    if n < 50:  # Need enough data for meaningful FFT
        return 0
        
    signal_fft = fft(signal)
    frequencies = fftfreq(n, d=1/sample_rate)
    
    positive_freqs = frequencies[:n//2]
    magnitudes = np.abs(signal_fft)[:n//2]
    
    # 0.5 Hz to 3.0 Hz (30 BPM - 180 BPM)
    valid_indices = np.where((positive_freqs >= 0.5) & (positive_freqs <= 3.0))
    valid_freqs = positive_freqs[valid_indices]
    valid_magnitudes = magnitudes[valid_indices]
    
    if len(valid_freqs) == 0:
        return 0
        
    dominant_freq = valid_freqs[np.argmax(valid_magnitudes)]
    return dominant_freq * 60

def process_oximeter_readings(red_data, ir_data, sample_rate):
    ros = calculate_ros(red_data, ir_data)
    spo2 = calculate_spo2(ros)
    heart_rate = calculate_heart_rate(ir_data, sample_rate)
    return spo2, heart_rate


# --- 3. UI Setup ---
root = tk.Tk()
root.title("ADVANCED SENSOR TELEMETRY")
root.geometry("500x800")  # Expanded size to fit new labels
root.configure(bg=BG_NORMAL)

# Global State
prox_history = deque(maxlen=BUFFER_SIZE)
red_history = deque(maxlen=PPG_BUFFER_SIZE)
ir_history = deque(maxlen=PPG_BUFFER_SIZE)
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
val_spo2 = create_label("BLOOD OXYGEN (SpO2 %)", TEXT_YELLOW)
val_hr = create_label("HEART RATE (BPM)", TEXT_PINK)

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
                
                # --- Alerts Logic ---
                pressure_low = p_val < THRESHOLD_PRESSURE
                
                if prox_val != -1:
                    prox_history.append(prox_val)
                    val_proximity.config(text=f"{int(prox_val)}")
                
                prox_danger = False
                if len(prox_history) == BUFFER_SIZE:
                    if all(x > THRESHOLD_PROX_HIGH for x in prox_history):
                        prox_danger = True

                is_alerting = pressure_low or prox_danger

                # --- Update Raw Sensors Displays ---
                val_pressure.config(text=f"{p_val:.2f}")
                val_red.config(text=f"{int(r_val)}")
                val_ir.config(text=f"{int(i_val)}")
                
                # --- SpO2 and Heart Rate Calculations ---
                # Check for low/bad signal (less than 2000)
                if r_val < MIN_SIGNAL_THRESH or i_val < MIN_SIGNAL_THRESH:
                    val_spo2.config(text="ERR")
                    val_hr.config(text="ERR")
                    # Clear histories so bad data doesn't skew subsequent FFTs
                    red_history.clear()
                    ir_history.clear()
                else:
                    # Accumulate valid data
                    red_history.append(r_val)
                    ir_history.append(i_val)
                    
                    # Wait until we have a substantial amount of data points before computing FFT
                    if len(red_history) >= 50: 
                        spo2, hr = process_oximeter_readings(list(red_history), list(ir_history), SAMPLE_RATE)
                        
                        # Only display if logic output sensible values
                        spo2_display = f"{spo2:.1f}" if spo2 > 0 else "--"
                        hr_display = f"{hr:.1f}" if hr > 0 else "--"
                        
                        val_spo2.config(text=spo2_display)
                        val_hr.config(text=hr_display)
                
            else:
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