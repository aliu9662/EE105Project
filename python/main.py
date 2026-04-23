import serial
import time
import tkinter as tk
from tkinter import font

# --- 1. Terminal Setup Phase ---
port = input("Enter the serial port (e.g., COM5 or /dev/ttyUSB0): ").strip()

try:
    # Small timeout so Tkinter doesn't freeze
    ser = serial.Serial(port, 115200, timeout=0.05) 
    time.sleep(2) # Give the Arduino time to reset
except Exception as e:
    print(f"Failed to connect to {port}. Error: {e}")
    exit()

print("="*50)
print("Ensure Module 3 is activated before proceeding.")
print("Activate the PPG sensor on the EE105 devboard using the command 'm3ppg'.")
print("="*50)

command = input("Type your command (>): ").strip()
ser.write((command + '\n').encode())
print("Launching UI Monitor...")

# --- 2. UI Window Phase ---
# Create the main window
root = tk.Tk()
root.title("Sensor Monitor")
root.geometry("400x350")
root.configure(bg="#222222") # Dark mode background

# Define a nice, large font
large_font = font.Font(family="Helvetica", size=24, weight="bold")

# Create labels for each sensor
lbl_pressure = tk.Label(root, text="Pressure: -- kPa", font=large_font, fg="#00FF00", bg="#222222")
lbl_pressure.pack(pady=15)

lbl_proximity = tk.Label(root, text="Proximity: --", font=large_font, fg="#CC00FF", bg="#222222")
lbl_proximity.pack(pady=15)

lbl_red = tk.Label(root, text="RED: --", font=large_font, fg="#FF3333", bg="#222222")
lbl_red.pack(pady=15)

lbl_ir = tk.Label(root, text="IR: --", font=large_font, fg="#33CCFF", bg="#222222")
lbl_ir.pack(pady=15)

def update_data():
    """Reads the serial port without blocking the UI"""
    try:
        latest_line = None
        
        # Drain the buffer line-by-line until we get to the most recent one
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                latest_line = line
                
        # If we successfully grabbed a new line, update the UI
        if latest_line:
            # Print raw data to the terminal so we can see what the Arduino is sending
            print(f"Raw received: {latest_line}") 

            # Expecting format: pressure,proximity,red,ir
            pressure_val, proximity_val, red_val, ir_val = map(float, latest_line.split(','))
            
            # Update Pressure, Red, and IR
            lbl_pressure.config(text=f"Pressure: {pressure_val:.2f} kPa")
            lbl_red.config(text=f"RED: {int(red_val)}")
            lbl_ir.config(text=f"IR: {int(ir_val)}")
            
            # Handle the Proximity Error state
            if proximity_val == -1:
                lbl_proximity.config(text="Proximity: Error")
            else:
                lbl_proximity.config(text=f"Proximity: {int(proximity_val)}")
            
    except ValueError as e:
        print(f"Format Error - Could not split into 4 numbers: {e}")
    except Exception as e:
        print(f"General Error: {e}")

    # Tell the window to run this function again in 50 milliseconds
    root.after(50, update_data)

def on_closing():
    """Ensures the serial port closes properly when you hit the X on the window"""
    if ser.is_open:
        ser.close()
        print("Serial port closed safely.")
    root.destroy()

# Map the "X" button on the window to our safe closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the continuous loop
root.after(50, update_data)

# THIS IS THE MAGIC LINE YOU WERE MISSING! 
# It keeps the window open until you click X.
root.mainloop()