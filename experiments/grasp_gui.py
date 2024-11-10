#!/bin/env python3
import tkinter as tk
from tkinter import DISABLED, NORMAL, ttk
from tkinter import messagebox
import time
from time import sleep
import numpy as np
import click
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import threading
from faive_system.src.hand_control.hand_controller import HandController

np.set_printoptions(precision=3, suppress=True)

"""
GUI for grasping objects control
options: --debug, --calibrate, --sim
"""
###################################################
#TODO: Just an example file, you need to adapt it to your hand
###################################################

grasp_type = 0

two_phase = False
grasp_duration = 2.0

goal_pos_grasp = [[0, 20, -38, 5, 0, 95, 110, 0, 95, 110, 0, 95, 110, 0, 95, 110], # Thumbs up!
                [30, -20, 50, 70, 20, 0, 0, -20, 0, 0, 0, 95, 110, 0, 95, 110], # Peace Sign
                [50, -20, 50, 50, 0, 80, 80, 0, 80, 80, 0, 80, 80, 0, 80, 80], # Power Grasp
                [70, 20, 20, 0, -15, 80, 30, -5, 80, 30, 5, 80, 30, 15, 80, 30], # Pinch Grasp
                [70, 20, 20, 0, -15, 80, 30, -5, 80, 30, 0, 0, 0, 0, 0, 0], # 2-Finger Pinch
                [40, -30, 80, 35, 0, 60, 50, 0, 60, 50, 0, 60, 50, 0, 60, 50,], #Large Diameter(Feix 1)
                [45, 20, -38, 5, 0, 95, 110, 0, 95, 110, 0, 95, 110, 0, 95, 110], # Light Tool (Feix 5)
                [30, -64, 44, 71, 23, 70, 80, -10, 70, 80, -30, 71, 80, 0, 95, 110], # Sphere 4 finger (Feix 26)
                [30, -64, 44, 71, 23, 70, 80, -21, 70, 80, 0, 0, 0, 0, 0, 0], # Sphere 3 finger (Feix 28)
                [70, -10, 110, 95, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45], # Extension Type (Feix 18)
                [0, -10, 40, 50, -8, 60, 50, 25, 30, 30, 0, 0, 0, 0, 0, 0], # Scissors Closed
                [50, -75, 30, 75, -25, 30, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Scissors Open
                [0, 0, 0, 0, -20, 10, 20, 20, 10, 20, 0, 0, 0, 0, 0, 0], # Adduction Grip (Feix 23)
                [0, -6, 55, 30, 15, 55, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Lateral(Feix 16)
                [0, -90, 30, 90, 30, 20, 30, 0, 0, 30, -30, 0, 55, -30, 0, 75], # Power Disk (Feix 10)
                [20, -70, 20, 45, 30, 30, 40, 10, 35, 45, -30, 95, 110, -30, 95, 110], # Tripod (Feix 14)
                [0, -80, 80, 25, -30, 50, 50, 0, 30, 30, 0, 30, 30, 0, 30, 30], # Palmar Pinch (Feix 9)
                ] 

goal_pos_dict = {
                "Thumbs Up": goal_pos_grasp[0],
                "Peace Sign": goal_pos_grasp[1],
                "Power Grasp": goal_pos_grasp[2],
                "Pinch Grasp": goal_pos_grasp[3],
                "2-Finger Pinch": goal_pos_grasp[4],
                "Large Diameter(Feix 1)": goal_pos_grasp[5],
                "Light Tool (Feix 5)": goal_pos_grasp[6],
                "Sphere 4 Finger (Feix 26)": goal_pos_grasp[7], 
                "Sphere 3 Finger (Feix 28)": goal_pos_grasp[8],
                "Extension Type (Feix 18)": goal_pos_grasp[9],
                "Scissors Closed (Feix 19)": goal_pos_grasp[10],
                "Scissors Open (Feix 19)": goal_pos_grasp[11],
                "Adduction Grip (Feix 23)": goal_pos_grasp[12],
                "Lateral(Feix 16)": goal_pos_grasp[13],
                "Power Disk (Feix 10)": goal_pos_grasp[14],
                "Tripod (Feix 14)": goal_pos_grasp[15],
                "Palmar Pinch (Feix 9)": goal_pos_grasp[16],
                }

class GraspGUI:
    def __init__(self, hc, debug=False, calibrate=False, sim=False):
        self.hc = hc
        self.windowopen = True
        self.debug = debug
        self.calibrate = calibrate
        self.sim = sim
        
        self.gui_root = tk.Tk()
        self.gui_root.geometry('1200x800')  # Increased width to accommodate the plot
        self.gui_root.resizable(True, True)
        self.gui_root.title('Joint-level control & grasping GUI')
        self.gui_root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.ctrl_mode = tk.IntVar(master=self.gui_root, value=0)
        self.grasp_selection = tk.StringVar(self.gui_root)
        self.grasp_selection.set(list(goal_pos_dict.keys())[0])
        
        self.joint_sliders_enabled = []
        self.joint_sliders = []
        self.joint_status_displays = []
        
        self.goal_current_slider = None
        self.goal_current_display = None
        self.goal_current_slider_enabled = False

        joint_nr = 0
        motor_nr = 0
        angles = []
        for mg in self.hc.muscle_groups:
            joint_nr += len(mg.joint_ids)
            motor_nr += len(mg.motor_ids)
            angles.extend(mg.calibration_max)

        # control input
        self.target_lock = threading.RLock()
        angles = np.array(angles, dtype=np.float64)
        self.joint_target_angles = np.zeros_like(angles, dtype=np.float64)
        # self.joint_target_angles = np.array(angles, dtype=np.float64)

        self.create_widgets()
        
        # Initialize plot data
        self.plot_window = 35  # seconds
        self.line_plot_data = {motor_id: deque(maxlen=self.plot_window * 10) for motor_id in self.hc.motor_ids}  # Assuming 10 data points per second
        self.dotten_line_plot_data = {motor_id: deque(maxlen=self.plot_window * 10) for motor_id in self.hc.motor_ids}  # Assuming 10 data points per second
        self.plot_times = deque(maxlen=self.plot_window * 10)
        self.start_time = time.time()
        
        self.update_plot()
        self.periodic_update()

    def create_widgets(self):
        self.gui_root.grid_columnconfigure(0, weight=3)
        self.gui_root.grid_columnconfigure(1, weight=1)
        self.gui_root.grid_rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.gui_root, padding="20 20 20 20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        plot_frame = ttk.Frame(self.gui_root, padding="10 20 20 20")
        plot_frame.grid(row=0, column=1, sticky="nsew")

        row_idx = 0

        # Control Mode Selection
        ttk.Label(main_frame, text="Control Mode", font=('Helvetica', 12, 'bold')).grid(row=row_idx, column=0, columnspan=2, pady=(0, 10))
        row_idx += 1

        self.radio_buttons = []
        self.radio_buttons.append(ttk.Radiobutton(main_frame, text="Joint Control", variable=self.ctrl_mode, value=0, command=self.change_ctrl_mode))
        self.radio_buttons[-1].grid(row=row_idx, column=0, padx=(20, 10), sticky="w")
        self.radio_buttons.append(ttk.Radiobutton(main_frame, text="Grasp Control", variable=self.ctrl_mode, value=1, command=self.change_ctrl_mode))
        self.radio_buttons[-1].grid(row=row_idx, column=1, padx=(10, 20), sticky="w")
        row_idx += 1

        ttk.Separator(main_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=15)
        row_idx += 1

        # Initialization and Reset buttons
        self.update_init_button = ttk.Button(main_frame, text="Update Initialization", command=self.update_initialization, width=25)
        self.update_init_button.grid(row=row_idx, column=0, padx=(20, 10), pady=(0, 15))
        self.reset_joints_button = ttk.Button(main_frame, text="Reset Joints", command=self.reset_joints, width=25)
        self.reset_joints_button.grid(row=row_idx, column=1, padx=(10, 20), pady=(0, 15))
        row_idx += 1

        # Grasp Control
        ttk.Label(main_frame, text="Grasp Control", font=('Helvetica', 12, 'bold')).grid(row=row_idx, column=0, columnspan=2, pady=(0, 10))
        row_idx += 1

        self.grasp_buttons = []
        self.grasp_buttons.append(ttk.Button(main_frame, text="Grasp", command=self.grasp, state=DISABLED, width=25))
        self.grasp_buttons[-1].grid(row=row_idx, column=0, padx=(20, 10))
        self.grasp_buttons.append(ttk.Button(main_frame, text="Release", command=self.release, state=DISABLED, width=25))
        self.grasp_buttons[-1].grid(row=row_idx, column=1, padx=(10, 20))
        row_idx += 1

        self.grasp_options = ttk.Combobox(main_frame, textvariable=self.grasp_selection, values=list(goal_pos_dict.keys()), state="readonly", width=30)
        self.grasp_options.grid(row=row_idx, column=0, columnspan=2, pady=(10, 0))
        self.grasp_options["state"] = DISABLED
        row_idx += 1

        ttk.Separator(main_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=15)
        row_idx += 1

        # Joint Control
        ttk.Label(main_frame, text="Joint Control", font=('Helvetica', 12, 'bold')).grid(row=row_idx, column=0, columnspan=2, pady=(0, 10))
        row_idx += 1

        joint_frame = ttk.Frame(main_frame)
        joint_frame.grid(row=row_idx, column=0, columnspan=2, sticky="nsew")
        joint_frame.grid_columnconfigure(1, weight=1)

        for muscle_group in self.hc.muscle_groups:
            ttk.Label(joint_frame, text=f"{muscle_group.name}", font=('Helvetica', 10, 'bold')).grid(row=row_idx, column=0, columnspan=3, pady=(10, 5), sticky="w")
            row_idx += 1
            for i, joint_id in enumerate(muscle_group.joint_ids):
                ttk.Label(joint_frame, text=f"Joint {joint_id}:").grid(row=row_idx, column=0, padx=(20, 10), sticky="w")
                range_min, range_max = muscle_group.joint_ranges[i]
                self.joint_sliders_enabled.append(False)
                slider = ttk.Scale(joint_frame, from_=range_min, to=range_max, command=lambda value, id=len(self.joint_sliders): self.slider_changed_cb(id, value), length=200)
                self.joint_sliders.append(slider)
                slider.grid(row=row_idx, column=1, padx=(0, 10), sticky="ew")
                self.joint_status_displays.append(ttk.Label(joint_frame, text="0.000", width=8))
                self.joint_status_displays[-1].grid(row=row_idx, column=2, padx=(0, 20), sticky="e")
                
                # Set the initial value
                self.joint_sliders[-1].set(muscle_group.calibration_max[i])
                self.joint_sliders_enabled[-1] = True
                row_idx += 1
        self.joint_sliders_enabled = np.array(self.joint_sliders_enabled)

        ttk.Separator(main_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=15)
        row_idx += 1

        ttk.Label(main_frame, text="Goal Current:").grid(row=row_idx, column=0, padx=(10, 10), sticky="w")
        range_min, range_max = muscle_group.joint_ranges[i]
        self.goal_current_slider_enabled = False
        self.goal_current_slider = ttk.Scale(main_frame, from_=0, to=750, command=lambda value: self.goal_current_slider_changed_cb(value), length=500)
        self.goal_current_slider.grid(row=row_idx, column=1, padx=(0, 20), sticky="ew")
        self.goal_current_display = ttk.Label(main_frame, text="0.000")
        self.goal_current_display.grid(row=row_idx, column=2, padx=(10, 10), sticky="e")
        self.goal_current_slider.set(0.0)
        self.goal_current_slider_enabled = True
        row_idx += 1

        # Create matplotlib figure and embed in Tkinter
        self.fig, self.ax = plt.subplots(figsize=(5, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ax.set_title("Motor Positions", fontsize=12)
        self.ax.set_xlabel("Time (s)", fontsize=12)
        self.ax.set_ylabel("Position (rad)", fontsize=12)
        self.lines = {}
        self.dotted_lines = {}
        for i, motor_id in enumerate(self.hc.motor_ids):
            line, = self.ax.plot([], [], label=str(motor_id), alpha=0.7)
            self.lines[motor_id] = line
            dotted_line, = self.ax.plot([], [], linestyle='--', color=line.get_color())
            self.dotted_lines[motor_id] = dotted_line

        self.ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize='small')
        self.ax.grid(True)
        self.ax.tick_params(axis='both', which='major', labelsize=10)

    def update_plot(self):
        current_time = time.time() - self.start_time
        self.plot_times.append(current_time)
        
        motor_positions = self.hc.get_motor_pos()
        for i, motor_id in enumerate(self.hc.motor_ids):
            self.line_plot_data[motor_id].append(motor_positions[i])
        
        with self.target_lock:
            motor_targets = self.hc.last_motor_pos_cmd.copy()
        for i, motor_id in enumerate(self.hc.motor_ids):
            self.dotten_line_plot_data[motor_id].append(motor_targets[i])

        x_min = max(0, current_time - self.plot_window)
        x_max = current_time
        self.ax.set_xlim(x_min, x_max)
        
        y_min = min(min(data) for data in self.line_plot_data.values()) - 0.1
        y_max = max(max(data) for data in self.line_plot_data.values()) + 0.1

        dotted_y_min = min(min(data) for data in self.dotten_line_plot_data.values()) - 0.1
        dotted_y_max = max(max(data) for data in self.dotten_line_plot_data.values()) + 0.1

        y_min = min(y_min, dotted_y_min)
        y_max = max(y_max, dotted_y_max)

        self.ax.set_ylim(y_min, y_max)
        
        for motor_id, line in self.lines.items():
            line.set_data(self.plot_times, self.line_plot_data[motor_id])
        for motor_id, dotted_line in self.dotted_lines.items():
            dotted_line.set_data(self.plot_times, self.dotten_line_plot_data[motor_id])
        self.canvas.draw()

    def periodic_update(self):
        if self.windowopen:
            self.gui_root.update()
            # self.update_plot()
            with self.target_lock:
                self.hc.command_joint_angles(self.joint_target_angles)
            self.gui_root.after(100, self.periodic_update)  # Update every 100ms
        else:
            self.gui_root.quit()

    def goal_current_slider_changed_cb(self, value):
        if not self.goal_current_slider_enabled:
            return

        value = float(value)
        with self.target_lock:
            motor_currents_mA = np.array([value] * len(self.hc.motor_ids))
            self.hc.write_desired_motor_current(motor_currents_mA)
            self.goal_current_display.config(text=f"{value:03.3f}")

    def slider_changed_cb(self, id, value):
        if not self.joint_sliders_enabled[id]:
            return

        value = float(value)
        with self.target_lock:
            self.joint_target_angles[id] = value
        self.joint_status_displays[id].config(text=f"{value:3.3f}")

    def change_ctrl_mode(self):
        """ Enables and Disables inputs according to control mode """
        if self.ctrl_mode.get() == 1: #if switched to grasp mode
            joint_mode = DISABLED
            grasp_mode = NORMAL
        else:
            joint_mode = NORMAL
            grasp_mode = DISABLED
        for i in range(len(self.joint_sliders)):
            self.joint_sliders[i]["state"]=joint_mode
        self.update_init_button["state"]=joint_mode
        self.reset_joints_button["state"]=joint_mode
        self.grasp_buttons[0]["state"]=grasp_mode
        self.grasp_buttons[1]["state"]=DISABLED
        self.grasp_options["state"]=grasp_mode

    def grasp(self):
        """ move hand to grasp position, en-/disable according inputs """
        goal_pos = goal_pos_dict[self.grasp_selection.get()]
        self.radio_buttons[0]["state"]=DISABLED
        self.radio_buttons[1]["state"]=DISABLED
        self.grasp_buttons[0]["state"]=DISABLED
        self.grasp_options["state"]=DISABLED
        if two_phase: #thumb after fingers
            with self.target_lock:
                currpos = self.move2pos_fingers(self.joint_target_angles, goal_pos, grasp_duration)
            self.move2pos_thumb(currpos, goal_pos, grasp_duration)
        else:
            with self.target_lock:
                self.move2pos(self.joint_target_angles,goal_pos,grasp_duration)
        self.grasp_buttons[1]["state"]=NORMAL

    def release(self):
        """ move hand to last joint position, en-/disable according inputs """
        self.grasp_buttons[1]["state"]=DISABLED    
        goal_pos = goal_pos_dict[self.grasp_selection.get()]
        if two_phase: #thumb after fingers
            with self.target_lock:
                currpos = self.move2pos_thumb(goal_pos, self.joint_target_angles, grasp_duration)
            self.move2pos_fingers(currpos,self.joint_target_angles,grasp_duration)
        else:
            with self.target_lock:
                self.move2pos(goal_pos,self.joint_target_angles,grasp_duration)
        self.grasp_buttons[0]["state"]=NORMAL
        self.radio_buttons[0]["state"]=NORMAL
        self.radio_buttons[1]["state"]=NORMAL   
        self.grasp_options["state"]=NORMAL

    def move2pos(self, startpos, goalpos, duration):
        ''' Input: starting position, desired position (joint angles in rad) and time for completion (seconds)
            Move to desired position within given time '''
        freq = 30 # Hz (max. 30 Hz)
        steps = int(freq*duration)
        currpos = startpos.copy()
        posdiff = goalpos - startpos
        for i in range(steps):
            currpos = startpos + posdiff/steps*(i+1.0)
            self.hc.command_joint_angles(currpos)
            sleep(1.0/freq-0.03)
        return currpos

    def move2pos_fingers(self, startpos, goalpos, duration):
        ''' Input: starting position, desired position (joint angles in rad) and time for completion (seconds)
            Move to desired position within given time '''
        freq = 30 # Hz (max. 30 Hz)
        steps = int(freq*duration)
        currpos = startpos.copy()
        posdiff = goalpos - startpos
        for i in range(steps):
            currpos[4:16] = startpos[4:16] + posdiff[4:16]/steps*(i+1.0)
            self.hc.command_joint_angles(currpos)
            sleep(1.0/freq-0.03)
        return currpos

    def update_initialization(self):
        # update initialization and set all joint sliders to 0:
        self.hc.update_motorinitpos()
        for i in range(len(self.joint_sliders)):
            self.joint_sliders[i].set(0)

    def reset_joints(self):
        #set all joint sliders to 0:
        for i in range(len(self.joint_sliders)):
            self.joint_sliders[i].set(0)

    def move2pos_thumb(self, startpos, goalpos, duration):
        ''' Input: starting position, desired position (joint angles in rad) and time for completion (seconds)
            Move to desired position within given time '''
        freq = 30 # Hz (max. 30 Hz)
        steps = int(freq*duration)
        currpos = startpos.copy()
        posdiff = goalpos - startpos
        for i in range(steps):
            currpos[0:4] = startpos[0:4] + posdiff[0:4]/steps*(i+1.0)
            self.hc.command_joint_angles(currpos)
            sleep(1.0/freq-0.03)
        return currpos

    def on_closing(self):
        print("Closing window")
        self.windowopen = False

@click.command()
@click.option('--debug', is_flag=True, default=False, help='Debug mode.')
@click.option('--calibrate', is_flag=True, default=False, help='Calibration mode')
@click.option('--sim', is_flag=True, default=False, help='use MuJoCo simulation')
def main(debug, calibrate, sim):
    if sim:
        hc = HandControllerMujocoSim()
    else:
        hc = HandController(port="/dev/ttyUSB0")
    hc.init_joints(calibrate=calibrate)

    gui = GraspGUI(hc, debug=debug, calibrate=calibrate, sim=sim)
    gui.gui_root.mainloop()
    
    hc.disable_torque()
    hc.disconnect_from_dynamixels()
    hc.stop()

    gui.windowopen = False
    gui.gui_root.quit()
    gui.gui_root.destroy()

if __name__ == "__main__":
    main()
