import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
import signal
import sys
from faive_system.src.hand_control import DynamixelIndirectClient, HardwareErrorStatus

np.set_printoptions(precision=3, suppress=True)

class MotorControllerApp:
    def __init__(self, root, motor_ids: list = list(range(1, 17)),
                 port: str = "",
                 baudrate: int = 3000000,
                 direction: list = []
                 ):
        self.root = root
        self.root.title("Motor Controller")
        self.root.geometry("1450x700")  # Increased width to accommodate new column
        
        self._dxc = DynamixelIndirectClient(motor_ids=motor_ids, port=port, baudrate=baudrate, lazy_connect=True)

        self.motor_lock = threading.Lock()
        self.motor_ids = np.array(motor_ids)
        self.motor_pos = np.zeros_like(self.motor_ids, dtype=np.float32)
        self.motor_target = np.zeros_like(self.motor_ids, dtype=np.float32)

        self.zero_pos = np.zeros_like(self.motor_ids, dtype=np.float32)
        
        if len(direction) == 0:
            direction = np.ones_like(self.motor_ids)
        self.direction = np.array(direction)

        self.status_lock = threading.Lock()

        self.motor_target_sliders = []
        self.slider_enabled = np.zeros_like(self.motor_ids, dtype=np.bool_)
        self.motor_status_displays = []
        self.motor_target_displays = []
        self.zero_pos_displays = []
        self.torque_vars = []
        self.torque_switches = []
        self.error_displays = []
        self.temp_displays = []
        self.current_displays = []
        self.tick_displays = []
        self.coupled_motion_vars = []  # New list to store the coupled motion variables

        self.goal_current = 200
        self.set_motor_desired_current(self.motor_ids, self.goal_current * np.ones_like(self.motor_ids))

        self.create_widgets()
        self.update_motor_pos()
        for idx, motor_id in enumerate(self.motor_ids):
            self.update_zero_pos(motor_id)
        self.root.after(100, self.periodic_update)
        self.root.after(100, self.check_hardware_status)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(main_frame, text="Motor Control", font=('Helvetica', 16, 'bold')).grid(row=0, column=0, columnspan=10, pady=10)

        headers = ['Motor ID', 'Coupled Motion', 'Position Control', 'Current Pos', 'Target Pos', 'Update Zero', 'Zero Pos', 'Torque', 'Reboot', 'Current', 'Temp', 'Tick', 'Errors']
        for col, header in enumerate(headers):
            ttk.Label(main_frame, text=header, font=('Helvetica', 12, 'bold')).grid(row=1, column=col, pady=5, padx=5)

        for idx, motor_id in enumerate(self.motor_ids):
            row = idx + 2
            ttk.Label(main_frame, text=f"{motor_id}", width=8, anchor="center").grid(row=row, column=0, pady=2, padx=5)

            coupled_var = tk.StringVar(value="0")
            coupled_box = ttk.Combobox(main_frame, textvariable=coupled_var, values=["-1", "0", "+1"], width=5, state="readonly")
            coupled_box.grid(row=row, column=1, pady=2, padx=5)
            self.coupled_motion_vars.append(coupled_var)

            pos_slider = ttk.Scale(main_frame, from_=-2*np.pi, to=2*np.pi, length=400,
                               command=lambda value, id=motor_id: self.pos_slider_changed_callback(id, value))
            pos_slider.grid(row=row, column=2, pady=2, padx=5)
            self.motor_target_sliders.append(pos_slider)

            status_display = ttk.Label(main_frame, text="0.000", width=8, anchor="center")
            status_display.grid(row=row, column=3, pady=2, padx=5)
            self.motor_status_displays.append(status_display)

            target_display = ttk.Label(main_frame, text="0.000", width=8, anchor="center")
            target_display.grid(row=row, column=4, pady=2, padx=10)
            self.motor_target_displays.append(target_display)

            zero_button = ttk.Button(main_frame, text="↻",
                                    command=lambda id=motor_id: self.update_zero_pos(id))
            zero_button.grid(row=row, column=5, pady=2, padx=5)

            zero_pos_display = ttk.Label(main_frame, text="0.000", width=8, anchor="center")
            zero_pos_display.grid(row=row, column=6, pady=2, padx=5)
            self.zero_pos_displays.append(zero_pos_display)

            torque_var = tk.BooleanVar(value=True)
            torque_switch = ttk.Checkbutton(main_frame, variable=torque_var, 
                                            command=lambda id=motor_id, var=torque_var: self.toggle_torque(id, var))
            torque_switch.grid(row=row, column=7, pady=2, padx=5)
            self.torque_vars.append(torque_var)
            self.torque_switches.append(torque_switch)

            reboot_button = ttk.Button(main_frame, text="Reboot",
                                       command=lambda id=motor_id: self.reboot_motor(id))
            reboot_button.grid(row=row, column=8, pady=2, padx=5)

            current_display = ttk.Label(main_frame, text="0.000", width=8, anchor="center")
            current_display.grid(row=row, column=9, pady=2, padx=5)
            self.current_displays.append(current_display)

            temp_display = ttk.Label(main_frame, text="0", width=8, anchor="center")
            temp_display.grid(row=row, column=10, pady=2, padx=5)
            self.temp_displays.append(temp_display)

            tick_display = ttk.Label(main_frame, text="0", width=8, anchor="center")
            tick_display.grid(row=row, column=11, pady=2, padx=5)
            self.tick_displays.append(tick_display)

            error_display = ttk.Label(main_frame, text="✓", width=8, anchor="center")
            error_display.grid(row=row, column=12, pady=2, padx=5)
            self.error_displays.append(error_display)

        update_all_button = ttk.Button(main_frame, text="Update All Zeros", command=self.update_all_zeros)
        update_all_button.grid(row=len(self.motor_ids)+2, column=0, columnspan=3, pady=10)

        write_offsets_button = ttk.Button(main_frame, text="CurrPos to EEPROM", command=self.write_ppos_to_eeprom)
        write_offsets_button.grid(row=len(self.motor_ids)+2, column=3, columnspan=3, pady=10)

        reboot_all_button = ttk.Button(main_frame, text="Reboot All Motors", command=self.reboot_all_motors)
        reboot_all_button.grid(row=len(self.motor_ids)+2, column=7, columnspan=3, pady=10)

        # Goal Current Slider
        ttk.Label(main_frame, text="Goal Current", font=('Helvetica', 12, 'bold')).grid(row=len(self.motor_ids)+3, column=0, columnspan=2, pady=10)
        self.goal_current_slider = ttk.Scale(main_frame, from_=0, to=750, length=400,
                                             command=self.goal_current_changed_callback)
        self.goal_current_slider.grid(row=len(self.motor_ids)+3, column=2, columnspan=5, pady=10)
        self.goal_current_display = ttk.Label(main_frame, text=f"{self.goal_current}", width=8)
        self.goal_current_display.grid(row=len(self.motor_ids)+3, column=7, columnspan=2, pady=10)
        self.goal_current_slider.set(self.goal_current)

    def pos_slider_changed_callback(self, motor_id, value):
        idx = np.where(self.motor_ids == motor_id)[0][0]
        if not self.slider_enabled[idx]:
            return

        # Compute delta for handling other (possibly) coupled motors
        delta = float(value) - (self.motor_target[idx] - self.zero_pos[idx])

        # Update the target position
        self.motor_target[idx] = self.direction[idx] * float(value) + self.zero_pos[idx]
        self.motor_target_displays[idx].config(text=f"{self.motor_target[idx]:.3f}")

        # Handle coupled motion
        for coupled_idx, coupled_var in enumerate(self.coupled_motion_vars):
            if coupled_idx != idx:  # Skip the motor that was directly moved
                coupled_value = coupled_var.get()
                if coupled_value != "0":
                    coupling_direction = 1 if coupled_value == "+1" else -1
                    self.slider_enabled[coupled_idx] = False
                    self.motor_target[coupled_idx] += self.direction[coupled_idx] * coupling_direction * delta
                    self.motor_target_sliders[coupled_idx].set(self.motor_target[coupled_idx] - self.zero_pos[coupled_idx])
                    self.slider_enabled[coupled_idx] = True

        self.update_motor_pos()
        self.update_motor_targets()

    def goal_current_changed_callback(self, value):
        self.goal_current = int(float(value))
        self.goal_current_display.config(text=f"{self.goal_current}")
        self.set_motor_desired_current(self.motor_ids, self.goal_current * np.ones_like(self.motor_ids))

    def set_motor_desired_current(self, motor_ids, curr_limits):
        # self._dxc.write_desired_current(motor_ids, curr_limits)
        # self._dxc.write_desired_current(motor_ids, curr_limits)
        self._dxc.gcurr = curr_limits

    def get_motor_pos(self):
        # return self._dxc.read_pos()
        return self._dxc.ppos
    
    def get_motor_goalpos(self):
        return self._dxc.gpos

    def get_motor_current(self):
        return self._dxc.pcurr
    
    def get_motor_temp(self):
        return self._dxc.ptemp

    def get_motor_ticks(self):
        return self._dxc.tick

    def update_motor_pos(self):
        with self.status_lock:
            self.motor_pos[:] = self.get_motor_pos()
            for idx, disp in enumerate(self.motor_status_displays):
                disp.config(text=f"{self.motor_pos[idx]:.3f}")

            self.motor_target_displays[idx].config(text=f"{self.motor_target[idx]:.3f}")

    def update_motor_targets(self):
        with self.status_lock:
            print(f"Target: {self.motor_target}")
            # self._dxc.write_desired_pos(self.motor_ids, self.motor_target)
            self._dxc.gpos = self.motor_target

    def update_zero_pos(self, motor_id):
        idx = np.where(self.motor_ids == motor_id)[0][0]
        self.slider_enabled[idx] = False
        self.zero_pos[idx] = self.motor_pos[idx]
        self.zero_pos_displays[idx].config(text=f"{self.zero_pos[idx]:.3f}")
        self.motor_target_sliders[idx].set(0)
        self.motor_target[idx] = self.zero_pos[idx]
        # self.motor_target_displays[idx].config(text=f"{self.motor_target[idx]:.3f}")
        self.slider_enabled[idx] = True
        # print(f"Zero position for motor {motor_id} updated to {self.zero_pos[idx]:.3f}")
        # print(f"Motor {motor_id} target position set to {self.motor_target[idx]:.3f}")
        # print(f"Motor {motor_id} current position set to {self.motor_pos[idx]:.3f}")

    def update_all_zeros(self):
        for motor_id in self.motor_ids:
            self.update_zero_pos(motor_id)
        print(f"Zero pos: {self.zero_pos}")

    def reboot_all_motors(self):
        self.reboot_motor(self.motor_ids)

    def write_ppos_to_eeprom(self):
        self.slider_enabled[:] = False
        
        # Clear homing offsets first to avoid mixing things up
        self._dxc.write_homing_offset(self.motor_ids, np.zeros_like(self.motor_ids))
        
        # Wait 100 ms for positions to update
        time.sleep(0.1)

        # Write current positions to EEPROM
        homing_offsets = -1 * self.get_motor_pos()
        self._dxc.write_homing_offset(self.motor_ids, homing_offsets)
        self.update_all_zeros()
        self.slider_enabled[:] = True

    def toggle_torque(self, motor_id, var):
        enabled = var.get()
        self._dxc.set_torque_enabled([motor_id], enabled)
        print(f"Motor {motor_id} torque {'enabled' if enabled else 'disabled'}")
        if enabled:
            idx = np.where(self.motor_ids == motor_id)[0][0]
            self.motor_target_sliders[idx].set(self.motor_pos[idx] - self.zero_pos[idx])

    def reboot_motor(self, motor_ids):
        print(f"Rebooting motor {motor_ids}")
        self._dxc.reboot(motor_ids)

    def check_hardware_status(self):
        # error_statuses = self._dxc.read_hardware_error_status()
        error_statuses = self._dxc.hwerror
        for idx, motor_id in enumerate(self.motor_ids):
            status = error_statuses[idx]
            if status == HardwareErrorStatus.NONE:
                self.error_displays[idx].config(text="√")
            else:
                error_emoji = "X"
                if status & HardwareErrorStatus.INPUT_VOLTAGE_ERROR:
                    error_emoji += "↯"
                if status & HardwareErrorStatus.OVERHEATING_ERROR:
                    error_emoji += "♨"
                if status & HardwareErrorStatus.ELECTRICAL_SHOCK_ERROR:
                    error_emoji += "▲"
                if status & HardwareErrorStatus.OVERLOAD_ERROR:
                    error_emoji += "+"
                self.error_displays[idx].config(text=error_emoji)

        temps = self.get_motor_temp()
        tick_values = self.get_motor_ticks()
        currents = self.get_motor_current()
        for idx, (current, temp, tick) in enumerate((zip(currents, temps, tick_values))):
            self.current_displays[idx].config(text=f"{currents[idx]:.3f}")
            self.temp_displays[idx].config(text=f"{temp}")
            self.tick_displays[idx].config(text=f"{tick}")

        self.root.after(100, self.check_hardware_status)

    def periodic_update(self):
        self.update_motor_pos()
        self.root.after(100, self.periodic_update)

    def cleanup(self):
        """Perform cleanup operations before shutting down."""
        print("Cleaning up...")
        # Disable torque on all motors
        self._dxc.set_torque_enabled(self.motor_ids, False)
        # Close the Dynamixel client connection
        self._dxc.stop()
        self._dxc.disconnect()
        print("Cleanup complete. Exiting.")
        self.root.quit()
        sys.exit(0)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--motors", required=True, help="Comma-separated list of motor IDs."
    )
    parser.add_argument(
        "-d", "--direction", default=None, help="Comma-separated list of motor winding directions."
    )
    parser.add_argument(
        "-p",
        "--port",
        default="",
        help="TTY port to connect to.",
    )
    parser.add_argument(
        "-b", "--baud", default=3000000, help="The baudrate to connect with."
    )
    parsed_args = parser.parse_args()
    motors = [int(motor) for motor in parsed_args.motors.split(",")]
    
    if parsed_args.direction is None:
        direction = [1] * len(motors)
    else:
        direction = [int(motor) for motor in parsed_args.direction.split(",")]
        direction = [-1 if d == -1 else 1 for d in direction]

    print(f"Args.motors: {motors}")

    is_direction_assumed_str = " (assumed)" if parsed_args.direction is None else ""
    print(f"Args.direction{is_direction_assumed_str}: {direction}")

    root = tk.Tk()
    app = MotorControllerApp(root, motors, parsed_args.port, parsed_args.baud, direction)
     
    def signal_handler(sig, frame):
        print("\nCtrl+C pressed. Initiating cleanup...")
        app.cleanup()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Initiating cleanup...")
    finally:
        app.cleanup()