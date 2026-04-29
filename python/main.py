import serial
import time
import tkinter as tk
from tkinter import font
from collections import deque
import numpy as np
from scipy.fft import fft, fftfreq
import pygame

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
        
    # AC Component: robust pulsatile amplitude estimate (less sensitive to drift/outliers)
    ac_red = np.percentile(red_signal, 95) - np.percentile(red_signal, 5)
    ac_ir = np.percentile(ir_signal, 95) - np.percentile(ir_signal, 5)
    
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
        
    signal_arr = np.array(signal, dtype=float)
    signal_centered = signal_arr - np.mean(signal_arr)
    signal_fft = fft(signal_centered)
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


# --- 3. warning system ---
class TelemetryAlertManager:
    def __init__(self, root, alert_label):
        self.root = root
        self.alert_label = alert_label
        
        # Initialize Audio
        pygame.mixer.init()
        self.snd_warn = pygame.mixer.Sound('beep_short.mp3')
        self.snd_crit = pygame.mixer.Sound('alarm_loud.mp3')
        
        self.current_level = 0  # 0: Stable, 1: Warning, 2: Critical
        self.flash_state = False
        
        # Start the internal flashing loop
        self._flash_loop()

    def evaluate_telemetry(self, pressure, prox_danger, spo2, hr):
        # 1. Check Critical Life-Safety Thresholds
        is_hypoxic = (0 < spo2 < 90.0) # Ignore 0 (which means ERR/No data)
        is_hr_abnormal = (0 < hr < 40) or (hr > 150)
        is_pressure_critical = pressure < THRESHOLD_PRESSURE
        
        if is_hypoxic or is_hr_abnormal or is_pressure_critical:
            self._set_level(2, self._get_critical_text(is_hypoxic, is_hr_abnormal, is_pressure_critical))
        # 2. Check Environmental Warnings
        elif prox_danger:
            self._set_level(1, "WARNING: PROXIMITY HAZARD")
        # 3. All Clear
        else:
            self._set_level(0, "SYSTEM STABLE")

    def _get_critical_text(self, hyp, hr_ab, press):
        errors = []
        if press: errors.append("LOW PRESSURE")
        if hyp: errors.append("HYPOXIA")
        if hr_ab: errors.append("HR ABNORMAL")
        return "CRITICAL: " + " | ".join(errors)

    def _set_level(self, level, text):
        if self.current_level != level:
            self.current_level = level
            pygame.mixer.stop() # Stop previous sounds
            
            if level == 0:
                self.alert_label.config(text=text, fg="#00FF41", bg=BG_NORMAL)
                self._apply_bg(BG_NORMAL)
            elif level == 1:
                self.alert_label.config(text=text, fg="black", bg="#FFD700")
                self._apply_bg("#443300")
                pygame.mixer.Sound.play(self.snd_warn, loops=-1)
            elif level == 2:
                self.alert_label.config(text=text, fg="white", bg="#FF0000")
                pygame.mixer.Sound.play(self.snd_crit, loops=-1)

    def _apply_bg(self, color):
        self.root.configure(bg=color)
        for widget in self.root.winfo_children():
            # Don't overwrite the alert label's specific background
            if widget != self.alert_label:
                widget.configure(bg=color)

    def _flash_loop(self):
        """Handles the pulsing visual effect for critical alerts"""
        if self.current_level == 2:
            self.flash_state = not self.flash_state
            current_bg = "#660000" if self.flash_state else "#220000"
            self._apply_bg(current_bg)
            
        self.root.after(400, self._flash_loop)


# --- 4. UI Setup ---
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
last_sample_ts = None
sample_interval_ms_history = deque(maxlen=50)

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
alert_manager = TelemetryAlertManager(root, lbl_alert)

val_pressure = create_label("ATMOSPHERIC PRESSURE (kPa)", TEXT_CYAN)
val_proximity = create_label("PROXIMITY VECTOR", "#CC00FF")
val_red = create_label("PPG RED CHANNEL", "#FF3333")
val_ir = create_label("PPG IR CHANNEL", "#33CCFF")
val_spo2 = create_label("BLOOD OXYGEN (SpO2 %)", TEXT_YELLOW)
val_hr = create_label("HEART RATE (BPM)", TEXT_PINK)

def update_data():
    global is_alerting, last_sample_ts, sample_interval_ms_history
    try:
        latest_line = None
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                latest_line = line
                
        if latest_line:
            if latest_line.count(',') == 3:
                parts = latest_line.split(',')
                p_val, prox_val, r_val, i_val = map(float, parts)
                now_ts = time.time()
                delta_ms = None if last_sample_ts is None else round((now_ts - last_sample_ts) * 1000.0, 2)
                last_sample_ts = now_ts
                if delta_ms is not None and delta_ms > 0:
                    sample_interval_ms_history.append(delta_ms)
                
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
                        effective_sample_rate = SAMPLE_RATE
                        if len(sample_interval_ms_history) >= 10:
                            median_interval_ms = float(np.median(sample_interval_ms_history))
                            if median_interval_ms > 0:
                                effective_sample_rate = 1000.0 / median_interval_ms

                        ros = calculate_ros(list(red_history), list(ir_history))
                        spo2, hr = process_oximeter_readings(
                            list(red_history),
                            list(ir_history),
                            effective_sample_rate,
                        )
                        
                        # Only display if logic output sensible values
                        spo2_display = f"{spo2:.1f}" if spo2 > 0 else "--"
                        hr_display = f"{hr:.1f}" if hr > 0 else "--"

                        alert_manager.evaluate_telemetry(p_val, prox_danger, spo2, hr)
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
root.mainloop()
