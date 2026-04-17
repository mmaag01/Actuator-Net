import numpy as np
import matplotlib.pyplot as plt

# profile generator/exporter class
# test log importer/exporter class
class Profile:
    # inputs: 
    # func_p: function gives p(t) with p in inc(?) and t in s
    # func_v: function gives v(t) with v in rad/s and t in s
    # func_t: function gives T(t) with T in Nm and t in s
    # t_range: tuple with start and end time
    # f_ctrl: control frequency / discretization step

    def __init__(self, t_range, ctrl_freq, func_v = np.nan, func_p = np.nan, func_t = np.nan, gains = [0.0, 0.0, 0.0], name = "dummy", folder = "\\"):
        print("Initializing Profile")
        self.folder = folder
        self.func_v = func_v
        self.func_t = func_t
        self.func_p = func_p
        self.ctrl_freq = ctrl_freq
        self.t_range = t_range
        self.gains = gains
        self.generate_series()
        self.name = name
    
    # Generate time series and corresponding velocity, position and torque profiles based on the input functions and time range
    def generate_series(self):
        self.times = np.arange(self.t_range[0], self.t_range[1], 1/self.ctrl_freq)
        try:
            #if velocity defined
            self.profile_v = self.func_v(self.times)
            print(self.profile_v)
            self.profile_p = np.cumsum(self.profile_v)*1/self.ctrl_freq
        except:
            #if position defined
            self.profile_p = self.func_p(self.times)
            self.profile_v = np.diff(self.profile_p)*self.ctrl_freq #convert to velocity
            self.profile_v = np.insert(self.profile_v, 0, self.profile_v[0]) #prepend first elem to maintain same length
        self.profile_t = self.func_t(self.times)
    
    # Plot the velocity, position and torque profiles
    # left y-axis: velocity profile (blue), position profile (red)
    # right y-axis: torque profile (green)
    # data is plotted as step function (not smooth), due to implementation of the motor controller
    def plot_profile(self, title = None):
        if title is None: title = self.name
        fig, ax = plt.subplots(1, figsize = (14, 8))
        ax.step(self.times, self.profile_v, label = 'Velocity Profile', c='b')
        ax.step(self.times, self.profile_p, label = 'Position Profile', c='r')
        plt.grid()
        ax.set_ylabel('Velocity (rad/s)', c='b')
        ax2 = ax.twinx()
        ax2.step(self.times, self.profile_t, label = 'Torque Profile', c='g')
        ax2.set_ylabel('Torque (Nm)', c='g')
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        ax.legend()

    def plot_subprofile(self, fig, axes, row, col, title = None, pos_sep = False, vel = True, tor = True):
        ax = axes[row, col]
        if pos_sep:
            ax.step(self.times, self.profile_p, label = 'Position Profile', c='r')
            plt.grid()
            ax.set_ylabel('Position (inc)', c='r')
            ax.set_ylim(-1, 1)
        elif vel:
            ax.step(self.times, self.profile_v, label = 'Velocity Profile', c='b')
            ax.step(self.times, self.profile_p, label = 'Position Profile', c='r')
            plt.grid()
            ax.set_ylabel('Velocity (rad/s)', c='b')
            ax.set_ylim(-13.0*1.1,13.0*1.1)
            if tor:
                ax2 = ax.twinx()
                ax2.step(self.times, self.profile_t, label = 'Torque Profile', c='g')
                ax2.set_ylabel('Torque (Nm)', c='g')
                ax2.set_ylim(-140*1.1,140*1.1)
        else:
            ax.step(self.times, self.profile_t, label = 'Torque Profile', c='g')
            ax.step(self.times, np.ones_like(self.times)*140, label = 'Torque Limit', c='k', ls='--')
            ax.step(self.times, np.ones_like(self.times)*-140, c='k', ls='--')
            plt.grid()
            ax.set_ylabel('Torque (Nm)', c='g')
            ax.set_ylim(-140*1.1,140*1.1)
        ax.set_xlabel('Time (s)')
        if title is not None:
            title += " ("
            if vel: title += str(np.round(np.max(self.profile_v), 2)) + "rad/s, "
            if tor: title += str(np.round(np.max(self.profile_t), 2)) + "Nm, "
            title += str(self.ctrl_freq) + "Hz)"
        ax.set_title(title)
        #ax.legend()
    # Write the velocity, position and torque profiles to a .mh file in the format expected by the motor controller, with unit conversions applied
    # rad/s to rpm, Nm to mNm with safety factor, time in ms
    def writeProfile(self):
        #define unit conversions
        safety_factor = 1.0
        rad_s_to_mrpm = 60000/(2*np.pi) # convert velocity from rad/s to mrpm
        rad_to_inc = 4096.0/(2*np.pi) # convert position from rad to inc, as expected by motor controller
        Nm_to_mNm = 1000*safety_factor # convert torque from Nm to mNm, with safety factor
        s_to_ms = 1000 # convert time from s to ms, as expected by motor controller
        
        opLen = len(self.times) # number of operation points, i.e., length of the profile
        tStep = np.mean(np.diff(self.times))*s_to_ms # time step in ms, as expected by motor controller
        timeStepStr = "#define timeStep_ms " + str((tStep)) # define time step in ms for motor controller
        runtime = opLen*tStep/s_to_ms # in seconds
        available_memory = 32763 # available memory in 32-bit values for motor controller
        requested_memory = opLen*3 # number of 32-bit values requested by the profiles for motor controller

        nOPsStr = "#define nOPs " + str(int(opLen)); # define number of operation points for motor controller
        KpStr = "#define Kp " + str(self.gains[0]) # define Kp gain for motor controller
        KiStr = "#define Ki " + str(self.gains[1]) # define Ki gain for motor controller
        KdStr = "#define Kd " + str(self.gains[2]) # define Kd gain for motor controller
        posStr = "long posOPs[" + str(opLen)+ "] = {" # define position operation points in inc for motor controller
        torStr = "long torOPs[" + str(opLen)+ "] = {" # define torque operation points in mNm for motor controller
        velStr = "long velOPs[" + str(opLen)+ "] = {" # define velocity operation points in mrpm for motor controller

        for element in self.profile_t: #torque (mNm) rounded to int, as expected by motor controller
            torStr += str(int(np.round(element*Nm_to_mNm))) + ","
        for element in self.profile_v: #velocity (mrpm) rounded to int, convert to rpm in ApossC as expected by motor controller
            velStr += str(int(np.round(element*rad_s_to_mrpm))) + ","
        for element in self.profile_p: #position (inc) and round to 3 decimal places, as expected by motor controller
            posStr += str(int(np.round(element*rad_to_inc))) + ","

        # Remove the trailing commas and add closing braces
        torStr, velStr, posStr = torStr[:-1], velStr[:-1], posStr[:-1]  # Remove the last comma
        torStr += "};"
        velStr += "};"
        posStr += "};"

        if requested_memory > available_memory:
            overshooot = requested_memory - available_memory
            t_overshoot = np.ceil(overshooot/3*tStep/s_to_ms) # in seconds
            print("Warning: Requested memory of " + str(requested_memory) + " 32-bit values exceeds available memory of " + str(available_memory) + " 32-bit values for motor controller. Please reduce runtime by " + str(t_overshoot) + " s.")
        
        # Write to .mh file
        # "C:\\Users\\maxma\\OneDrive\\ETH\\Bachelors_Thesis\\Data Collection\\Aposs Code\\.mh\\Torque Step"
        with open(("C:\\Users\\maxma\\OneDrive\\ETH\\Bachelors_Thesis\\Data Collection\\Aposs Code\\.mh\\"+self.folder+self.name + ".mh"), 'w') as f:
            f.write("// "+ self.name + " " + str(self.ctrl_freq) + "Hz" + str(len(self.times)) + "pts.mh \n")
            f.write("// Requested 32 bit memory: " + str(requested_memory) + "/" + str(available_memory) + "\n")
            f.write("// Total Runtime: " + str(runtime) + " s \n")
            f.write(timeStepStr + "\n")
            f.write(nOPsStr + "\n")
            f.write(KpStr + "\n")
            f.write(KiStr + "\n")
            f.write(KdStr + "\n")
            f.write(torStr + "\n")
            f.write(velStr + "\n")
            f.write(posStr + "\n")