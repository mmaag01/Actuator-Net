import os
from os import times
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
import pygwalker as pyg
from scipy import signal, interpolate
from scipy.signal import find_peaks
from scipy.fftpack import fft
import scipy.io.wavfile as wavfile
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d

def ramp_func(x0, x1, times):
    x = np.zeros_like(times)
    x[0] = x0
    x[-1] = x1
    for i in range(1, len(times)-1):
        x[i] = x0 + (x1 - x0) * (times[i] - times[0]) / (times[-1] - times[0])
    return x

def ramp_func_dx(x0, x1, times):
    x = np.zeros_like(times)
    x[0], x[-1] = x0, x1
    t0, t1 = times[0], times[-1]
    T = t1-t0
    for i in range(1, len(times)-1):
        t = times[i]-t0
        x[i] = x0*t + (x1 - x0) * t**2 / (2*T)
    return x

class FreqFunctions:
    ms_to_s = 0.001
    rpm_to_rad_s = 2.0 * np.pi / 60.0
    def __init__(self, freq_range = [0.5, 25.0], seed=42, velmax=13.0, max_torque=140, RandomPhase=True, PosOffset=0.0, TorqueOffset=0.0, tRamp=0.0):
        self.freq_range = freq_range
        self.velmax = velmax
        self.max_torque = max_torque
        self.seed = seed
        self.PosOffset = PosOffset
        self.TorqueOffset = TorqueOffset
        self.tRamp = tRamp
        self.RandomPhase = RandomPhase

    def __repr__(self):
        return f"fMin={self.freq_range[0]} Hz, fMax={self.freq_range[1]} Hz, RandomPhase={self.RandomPhase}, Max. Velocity={self.velmax} rad/s, Max. Torque={self.max_torque} Nm, PosOffset={self.PosOffset}, TorqueOffset={self.TorqueOffset} Nm, Ramp Time={self.tRamp}s)"

    def _periodic_frequencies(self, rep_dur, n_sin):
        fMin, fMax = self.freq_range[0], self.freq_range[1]
        n_min = int(np.ceil(fMin * rep_dur))
        n_max = int(np.floor(fMax * rep_dur))
        if n_max < n_min: raise ValueError("No periodic frequencies available in the requested range.")
        m_vals = self.rng.integers(n_min, n_max + 1, size=n_sin)
        return m_vals / rep_dur

    def PosChirp(self, times, velmax=None):
        if velmax is None: velmax = self.velmax
        MaxAmp = velmax/(2*np.pi*self.freq_range[1]) # incs
        #print(f"Function: Linear Position Chirp,  Duration={np.max(times)*self.ms_to_s}s")

        t= times/self.ms_to_s
        Tms = np.max(t) # ms
        Ts = Tms*self.ms_to_s # s
        
        fMin, fMax = self.freq_range[0], self.freq_range[1]
        Amp = np.ones(len(t))*MaxAmp #define ref amplitude
        Theta = 2.0*np.pi*(fMin*t + t**2 * (fMax-fMin)/(2*Ts))

        PosRef = self.PosOffset + Amp * np.sin(Theta)

        return PosRef

    def LogTorChirp(self, times, tormax=None, freq_range = None):
        if freq_range is None: freq_range = self.freq_range
        if tormax is None: tormax = self.max_torque
        ms_to_s = 0.001
        
        ChirpTorqueAmpl = tormax
        ChirpFreqStart = freq_range[0] # Hz
        ChirpFreqFinal = freq_range[1] # Hz
        
        AmplRampUpTime = 0.0 # ms
        
        t= times/ms_to_s
        ChirpDuration = np.max(t) # ms
        
        ChirpFreqScale = ChirpFreqFinal/ChirpFreqStart
        ChirpInstFreqStart = ChirpFreqStart*ChirpDuration*ms_to_s/np.log(ChirpFreqScale)
        ChirpInstFreqFinal = ChirpFreqScale*ChirpInstFreqStart
        ChirpFreqBase = pow(ChirpFreqScale, (1.0/ChirpDuration))
        InstFreqCorrFactor = np.log(ChirpFreqScale)*1000/(ms_to_s*ChirpDuration)
        PhaseCorrection = 2.0*np.pi*ChirpInstFreqStart
        
        ChirpTime = ChirpDuration + AmplRampUpTime  #duration of chirp
        
        ChirpTorqueAmplRampUp = np.ones(len(t))*ChirpTorqueAmpl #define ref amplitude
        
        ChirpTorqueAmplRampUp[t < AmplRampUpTime] *= (t[t < AmplRampUpTime]/AmplRampUpTime) #ramp up ref amplitude
        
        ChirpFreq = ChirpInstFreqStart * np.power(ChirpFreqBase, t)
        TorqueRef =  ChirpTorqueAmplRampUp * np.sin(2.0*np.pi*ChirpFreq - PhaseCorrection) 

        return TorqueRef
    
    def LogPosChirp(self, times, scale = 1.0, velmax=None, freq_range = None, t_ramp = 0.1, vel = False):
        if freq_range is None: freq_range = self.freq_range
        if velmax is None: velmax = self.velmax
        MaxAmp = velmax/(2*np.pi*self.freq_range[1]) # incs
        #print(f"Function: Logarithmic Position Chirp, Duration={np.max(times)*self.ms_to_s}s")
        Vel = np.zeros_like(times, dtype=float)
        Pos = np.zeros_like(times, dtype=float)

        t = (times)[times <= times[-1]-t_ramp]/self.ms_to_s # ms
        t_ramp /= self.ms_to_s
        Tms = np.max(t) # ms
        print(Tms, t_ramp, times[-1]/self.ms_to_s)
        Ts = Tms*self.ms_to_s # s
        fStart, fFinal = freq_range[0], freq_range[1]
        fScale = fFinal/fStart
        fInstStart = fStart*Ts/np.log(fScale)
        fInstFinal = fScale*fInstStart
        fBase = pow(fScale, (1.0/Tms))
        fInstCorrFactor = np.log(fScale)/Ts
        PhaseCorr = 2.0*np.pi*fInstStart
        f = fInstStart * np.power(fBase, t)
        
        RefAmp = np.ones(len(t)) #define ref amplitude
        #Amp[t < self.tRamp] *= (t[t < self.tRamp]/self.tRamp) #ramp up ref amplitude
        
        Theta = 2.0*np.pi*fInstStart*np.power(fBase, t)
        dTheta_dt = Theta * fInstCorrFactor

        PosRef = RefAmp * np.sin(Theta - PhaseCorr)
        VelRef = RefAmp * dTheta_dt * np.cos(Theta - PhaseCorr)

        MaxAmp = np.max(np.abs(VelRef))
        Amp = velmax/MaxAmp*RefAmp

        PosRef *= Amp
        VelRef *= Amp

        VelFin, PosFin = VelRef[-1], PosRef[-1]
        Vel[times <= Ts], Vel[times > Ts] = VelRef, ramp_func(VelFin, 0.0, times[times > Ts])
        Pos[times <= Ts], Pos[times > Ts] = PosRef, ramp_func_dx(VelFin, 0.0, times[times > Ts])+PosFin

        #print(f"Max Position Reference = {np.max(np.abs(PosRef))} incs, Max Velocity Reference = {np.max(np.abs(VelRef))} rad/s")

        if vel is True:
            return Vel
        else:
            return Pos

    # Generate one periodic torque multisine over a single interval
    def TorqueMultisineGenRaw(self, times, n_sin, rep_dur, freq_range, max_torque, scaled):
        times = np.asarray(times)
        delta_t = times - times[0]  # local time within this interval

        # Allowed periodic frequencies: f = m / T with integer m
        n_min = int(np.ceil(freq_range[0] * rep_dur))
        n_max = int(np.floor(freq_range[1] * rep_dur))
        if n_max < n_min:
            raise ValueError("No periodic frequencies available in the requested range for this interval length.")

        m_vals = self.rng.integers(n_min, n_max + 1, size=n_sin)
        f = m_vals / rep_dur

        if self.RandomPhase:
            phase = self.rng.uniform(0.0, 2.0 * np.pi, size=n_sin)
        else:
            phase = np.zeros(n_sin)

        # 1/sqrt(f) weighting
        amp_shape = 1.0 / np.sqrt(f)
        amp = amp_shape / np.sum(amp_shape)
        amp *= max_torque

        tau_out = np.zeros_like(times, dtype=float)
        for i in range(n_sin):
            tau_out += amp[i] * np.sin(2.0 * np.pi * f[i] * delta_t + phase[i])
            tau_out -= amp[i] * np.sin(phase[i])

        tau_max = np.max(np.abs(tau_out))
        if scaled and tau_max > 0:
            scale_factor = max_torque / tau_max
            tau_out *= scale_factor
            amp *= scale_factor
            tau_max = np.max(np.abs(tau_out))
            
        return tau_out, f, amp, phase, np.max(np.abs(tau_out))
    # Generate a full torque multisine experiment with a new random multisine every interval
    def TorqueMultisineRaw(self, times, rep_dur, n_sin, t_ramp = 0.1, seed=None, freq_range = None, max_torque=None, scaled = False):
        if seed is None: seed = self.seed
        if max_torque is None: max_torque = self.max_torque
        if freq_range is None: freq_range = self.freq_range
        self.rng = np.random.default_rng(seed)

        print(f"Function: Torque Multisine, Seed = {seed}, Freq Range = {freq_range} Hz, Sinuses = {n_sin}, Duration = {times[-1]}, Full Reps = {np.floor(times[-1]/rep_dur)} (+1 Half-Rep), Time/Rep = {rep_dur}s")

        times = np.asarray(times)
        t = times[times <= times[-1]-t_ramp]
        reps = np.ceil(t[-1]/rep_dur)

        tau_out = np.zeros_like(t, dtype=float)
        Tau = np.zeros_like(times, dtype=float)

        total_duration = min(reps * rep_dur, t[-1] + (t[1] - t[0]))
        mask_total = (t >= 0.0) & (t < total_duration)

        for j in range(int(reps)):
            t0 = j * rep_dur
            t1 = (j + 1) * rep_dur
            mask = mask_total & (t >= t0) & (t < t1)

            if np.any(mask):
                tau_out[mask], freqs, amps, phase, max_amp = self.TorqueMultisineGenRaw(
                    t[mask], n_sin, rep_dur, freq_range, max_torque, scaled
                    )
                print("Interval {}: Frequencies = {}, Amplitudes = {}, Phases = {}, Max Amplitude = {}".format(j+1, freqs, amps, phase, max_amp))
        tauFin = tau_out[-1]
        Tau[times < total_duration], Tau[times >= total_duration] = tau_out, ramp_func(tauFin, 0.0, times[times >= total_duration])
        return Tau

    def PosMultisineGen(self, interval, n_sin, rep_dur, freq_range, velmax, scaled):
        interval = np.asarray(interval)
        tau = interval - interval[0]  # local time within this interval

        # Allowed periodic frequencies: f = m / T with integer m
        n_min = int(np.ceil(freq_range[0] * rep_dur))
        n_max = int(np.floor(freq_range[1] * rep_dur))
        if n_max < n_min:
            raise ValueError("No periodic frequencies available in the requested range for this interval length.")

        m_vals = self.rng.integers(n_min, n_max + 1, size=n_sin)
        f = m_vals / rep_dur
        if self.RandomPhase: phase = self.rng.uniform(0.0, 2.0 * np.pi, size=n_sin)
        else: phase = np.zeros(n_sin)

        A = velmax / (2.0 * np.pi * f * n_sin)

        pos_out = np.zeros_like(interval, dtype=float)
        vel_out = np.zeros_like(interval, dtype=float)
        for i in range(n_sin):
            theta = 2.0 * np.pi * f[i] * tau
            pos_out += A[i] * np.sin(theta + phase[i])
            pos_out -= A[i] * np.sin(phase[i])
            vel_out += A[i] * 2.0 * np.pi * f[i] * np.cos(theta + phase[i])
        velmax_actual = np.max(np.abs(vel_out))
        if scaled and velmax_actual > 0:
            scale_factor = velmax / velmax_actual
            pos_out *= scale_factor
            vel_out *= scale_factor
            A *= scale_factor
            velmax_actual = np.max(np.abs(vel_out))
        return pos_out, vel_out, f, A, phase, np.max(np.abs(vel_out))
    # Generate a full position multisine experiment with a new random multisine every interval
    def PosMultisine(self, times, rep_dur, n_sin, t_ramp = 0.1, seed=None, freq_range = None, velmax=None, scaled = False, vel = False):
        if seed is None: seed = self.seed
        if velmax is None: velmax = self.velmax
        if freq_range is None: freq_range = self.freq_range
        self.rng = np.random.default_rng(seed)
        print(velmax)
        
        print(f"Function: Position Multisine, Seed = {seed}, Freq. Range = {freq_range} Hz, Sinuses = {n_sin}, Duration = {times[-1]}, Full Reps = {np.floor(times[-1]/rep_dur)} (+1 Half-Rep), Time/Rep = {rep_dur}s, Max. Dur = {times[-1]}s")
        
        times = np.asarray(times)
        t = times[times <= times[-1]-t_ramp]
        reps = np.ceil(t[-1]/rep_dur)

        pos_out = np.zeros_like(t, dtype=float)
        vel_out = np.zeros_like(t, dtype=float)
        Pos, Vel = np.zeros_like(times, dtype=float), np.zeros_like(times, dtype=float)
        
        rad_to_rpm = 60/(2*np.pi)
        total_duration = min(reps * rep_dur, t[-1] + (t[1] - t[0]))
        mask_total = (t >= 0.0) & (t < total_duration)
        
        for j in range(int(reps)):
            t0 = j * rep_dur
            t1 = (j + 1) * rep_dur
            mask = mask_total & (t >= t0) & (t < t1)

            if np.any(mask):
                pos_out[mask], vel_out[mask], f, A, phase, velrad_s = self.PosMultisineGen(
                    t[mask], n_sin, rep_dur, freq_range, velmax, scaled
                    )
                print("Interval {}: Frequencies = {}, Amplitudes = {}, Phases = {}, Max Abs Vel (rad/s) = {}, Max Abs Vel (rpm) = {}".format(j+1, f, A, phase, velrad_s, velrad_s*rad_to_rpm))
                print(pos_out[mask].ndim, pos_out[mask].shape)
                if velrad_s > velmax:
                    print("Warning: Max Abs velocity {} rad/s ({} rpm) exceeds velmax {} ({} rpm) rad/s.".format(velrad_s, velrad_s*rad_to_rpm, velmax, velmax*rad_to_rpm))
        
        PosFin, VelFin = pos_out[-1], vel_out[-1]
        print(PosFin, VelFin)
        Pos[times < total_duration], Pos[times >= total_duration] = pos_out, ramp_func_dx(VelFin, 0, times[times >= total_duration]) + PosFin
        Vel[times < total_duration], Vel[times >= total_duration] = vel_out, ramp_func(VelFin, 0, times[times >= total_duration])

        if vel is True: return Vel
        else: return Pos
        
    def _smooth_sign_changes(self, u, times, trans_width=0.02):
        """
        Smooth only around zero crossings using a half cosine blend.
        u : array_like
            Torque command [Nm]
        times : array_like
            Time vector [s]
        trans_width : float
            Total smoothing width around each zero crossing [s]
        """
        u = np.asarray(u, dtype=float).copy()
        times = np.asarray(times, dtype=float)

        if len(u) < 3: return u

        half_w = 0.5 * trans_width
        signs = np.sign(u)

        # treat exact zeros robustly
        for i in range(1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i - 1]
        for i in range(len(signs) - 2, -1, -1):
            if signs[i] == 0:
                signs[i] = signs[i + 1]

        cross_idx = np.where(signs[:-1] * signs[1:] < 0)[0]

        for idx in cross_idx:
            t_cross = 0.5 * (times[idx] + times[idx + 1])
            mask = (times >= t_cross - half_w) & (times <= t_cross + half_w)
            if np.count_nonzero(mask) < 3:
                continue

            i0 = np.where(mask)[0][0]
            i1 = np.where(mask)[0][-1]

            y0 = u[i0]
            y1 = u[i1]
            tau = (times[mask] - times[i0]) / (times[i1] - times[i0])

            # half cosine interpolation: C1 smooth
            blend = 0.5 * (1.0 - np.cos(np.pi * tau))
            u[mask] = (1.0 - blend) * y0 + blend * y1

        return u
    
    def _slew_limit(self, u, dt, max_slope):
        """Limit du/dt to +/- max_slope [Nm/s]."""
        u = np.asarray(u, dtype=float)
        y = np.empty_like(u)
        y[0] = u[0]

        max_step = max_slope * dt
        for k in range(1, len(u)):
            du = u[k] - y[k - 1]
            du = np.clip(du, -max_step, max_step)
            y[k] = y[k - 1] + du
        return y

    def TorqueMultisineGen(self,times,n_sin,rep_dur,max_torque,max_slope=None,smooth=False,width=0.02,renorm=False):
        """
        Generate one periodic torque multisine over a single interval.
        max_slope : float or None
            Max torque slope [Nm/s]. If None, no slew limit is applied.
        smooth : bool
            If True, smooth only around sign changes.
        width : float
            Width of smoothing window around zero crossings [s]
        renorm : bool
            If True, rescales final signal back to max_torque.
            Usually keep this False, otherwise you partly undo the smoothing.
        """
        times = np.asarray(times, dtype=float)
        delta_t = times - times[0]
        # periodic frequencies f = m / T
        n_min = int(np.ceil(self.freq_range[0] * rep_dur))
        n_max = int(np.floor(self.freq_range[1] * rep_dur))
        if n_max < n_min:
            raise ValueError(
                "No periodic frequencies available in the requested range for this interval length."
            )

        m_vals = self.rng.integers(n_min, n_max + 1, size=n_sin)
        f = m_vals / rep_dur

        if self.RandomPhase:
            phase = self.rng.uniform(0.0, 2.0 * np.pi, size=n_sin)
        else:
            phase = np.zeros(n_sin)

        # 1/sqrt(f) weighting
        amp_shape = 1.0 / np.sqrt(f)
        amp = amp_shape / np.sum(amp_shape)

        tau_out = np.zeros_like(times, dtype=float)
        for i in range(n_sin):
            tau_out += amp[i] * np.sin(2.0 * np.pi * f[i] * delta_t + phase[i])
            tau_out -= amp[i] * np.sin(phase[i])

        # normalize raw multisine first
        peak = np.max(np.abs(tau_out))
        if peak > 0:
            tau_out = max_torque * tau_out / peak
            amp = max_torque * amp / peak
        else:
            amp = max_torque * amp

        # optional local smoothing near sign changes
        if smooth:
            tau_out = self._smooth_sign_changes(
                tau_out,
                times,
                trans_width=width
            )

        # optional slew rate limit
        if max_slope is not None:
            if len(times) < 2:
                raise ValueError("times must contain at least two samples.")
            dt = times[1] - times[0]
            tau_out = self._slew_limit(tau_out, dt=dt, max_slope=max_slope)

        # optional torque offset
        tau_out = tau_out + self.TorqueOffset

        # make sure hard limit is still respected
        final_peak = np.max(np.abs(tau_out))
        if final_peak > max_torque:
            if renorm:
                tau_out = max_torque * tau_out / final_peak
            else:
                tau_out = np.clip(tau_out, -max_torque, max_torque)

        return tau_out, f, amp, np.max(np.abs(tau_out))

    def TorqueMultisine(self,times,reps,rep_dur,n_sin,seed=None,max_torque=None,max_slope=None,smooth=False,width=0.02,renorm=False):
        if max_torque is None:
            max_torque = self.max_torque
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)

        print(
            f"Function: Torque Multisine, Freqs = {n_sin}, "
            f"Duration = {reps * rep_dur}s, "
            f"Reps = {reps}, Time/Rep = {rep_dur}s"
        )

        times = np.asarray(times, dtype=float)
        tau_out = np.zeros_like(times, dtype=float)

        total_duration = min(reps * rep_dur, times[-1] + (times[1] - times[0]))
        mask_total = (times >= 0.0) & (times < total_duration)

        for j in range(reps):
            t0 = j * rep_dur
            t1 = (j + 1) * rep_dur
            mask = mask_total & (times >= t0) & (times < t1)

            if np.any(mask):
                tau_seg, freqs, amps, max_amp = self.TorqueMultisineGen(
                    times[mask],
                    n_sin=n_sin,
                    rep_dur=rep_dur,
                    max_torque=max_torque,
                    max_slope=max_slope,
                    smooth=smooth,
                    width=width,
                    renorm=renorm
                )
                tau_out[mask] = tau_seg
                print(
                    "Interval {}: Frequencies = {}, Amplitudes = {}, Max Amplitude = {}".format(
                        j + 1, freqs, amps, max_amp
                    )
                )

        return tau_out
    
def _add_ramped_pulse(t_local, t_0, tor, t_hold, t_ramp):
    """
    Helper for one ramped torque pulse:
    0 -> amp -> 0

    Parameters
    ----------
    t_local : Local times vector (s)
    t_0 : Start time of the pulse (s).
    tor : Pulse amplitude in Nm.
    t_hold : Hold time at full amplitude (s).
    t_ramp : Ramp up/down time (s).

    Returns
    -------
    y : np.ndarray
        Pulse contribution over t_local.
    end_time : float
        End time of the full pulse.
    """
    y = np.zeros_like(t_local, dtype=float)

    t1 = t_0
    t2 = t1 + t_ramp
    t3 = t2 + t_hold
    t4 = t3 + t_ramp

    if t_ramp > 0:
        # ramp up
        m = (t_local >= t1) & (t_local < t2)
        y[m] = tor * (t_local[m] - t1) / t_ramp
        #print(f"Ramped from {np.min(y[m])} to {np.max(y[m])} Nm for ",t2-t1, " s")

        # hold
        m = (t_local >= t2) & (t_local < t3)
        y[m] = tor
        #print(f"Held at {tor} Nm for ", t3-t2, " s")

        # ramp down
        m = (t_local >= t3) & (t_local < t4)
        y[m] = tor * (1.0 - (t_local[m] - t3) / t_ramp)
        #print(f"Ramped down from {np.max(y[m])} to {np.min(y[m])} Nm for ",t4-t3, " s")
    else:
        # ideal step pulse
        m = (t_local >= t1) & (t_local < t3)
        y[m] = tor

    return y, t4

def torqueStep(times, torque_seq, hold_seq, cool_seq, max_torque=140.0, reps=1, ramp_times=None):
    """
    0 -> +tau -> 0 -> cool -> 0 -> -tau -> 0 -> cool
    Tor (Nm)    Hold (s)    Cool (s)    Ramp (s)
    35          0.2         0.5         0.09
    70          0.3         1.5         0.18
    105         0.2         1.5         0.27
    140                                 0.36
    """
    times = np.asarray(times, dtype=float)
    torque_seq = np.asarray(torque_seq, dtype=float)
    hold_seq = np.asarray(hold_seq, dtype=float)
    cool_seq = np.asarray(cool_seq, dtype=float)

    assert len(torque_seq) == len(hold_seq) == len(cool_seq), \
        "torque_seq, hold_seq, and cool_seq must have the same length."
    assert np.max(np.abs(torque_seq)) <= max_torque, \
        "Requested torque exceeds max_torque."
    assert np.all(np.array(ramp_times) >= 0.0), "ramp_times must be nonnegative."
    assert reps >= 1, "reps must be at least 1."

    # total duration of one full interval
    T_tot = 2 * (sum(hold_seq) + sum(cool_seq) + 2.0 * sum(ramp_times))

    #assert times[-1] >= reps * T_tot + 1e-12, \
    #    "torque duration exceeds max. permitted time."

    tau = np.zeros_like(times, dtype=float)

    for j in range(reps):
        t_shift = j * T_tot
        t_local = times - t_shift
        cursor = 0.0
        #print(f"Starting Torque Step Interval {j+1}/{reps}")

        for torque_des, hold, cool, ramp_time in zip(torque_seq, hold_seq, cool_seq, ramp_times):
            # positive pulse
            y, cursor = _add_ramped_pulse(t_local, cursor, torque_des, hold, ramp_time)
            tau += y
            cursor += cool

            # negative pulse
            y, cursor = _add_ramped_pulse(t_local, cursor, -torque_des, hold, ramp_time)
            tau += y
            cursor += cool

    return tau

def vel_tor_profile(times, n_vel=20, t_ramp=1.5):
    times = np.asarray(times)
    t0, t1 = times[times < t_ramp], times[times >= t_ramp]
    it0, it1 = int(len(t0)), int(len(t1))
    vel, tor = np.zeros_like(times), np.zeros_like(times)
    vel_ref, tor_ref = np.zeros_like(t1), np.zeros_like(t1)
    vel[:it0] = ramp_func(0.0, 13.0, t0)

    vel_boundary = np.array([13.0, 12.0, 10.5, 6.0, 0.0])
    tor_boundary = np.array([0.0, 50.0, 100.0, 135.0, 140.0])
    # interpolate max torque vs velocity
    tor_func = interp1d(vel_boundary, tor_boundary, kind='cubic', fill_value='extrapolate')
    # create a slow sweep over velocity
    vel_ref = np.linspace(vel_boundary.max(), 0, len(t1))
    # corresponding max torque
    tor_ref = tor_func(vel_ref)

    # sweep torque inside limits (e.g. sinusoidal)
    vel[it0:] = vel_ref
    tor[it0:] = tor_ref
    print(len(vel), len(tor), len(times))
    return vel, tor

# Constant signal of amplitude with same shape as input x
def constant_func(x, amplitude):
    return np.full_like(x, amplitude)

# Define a ramp function
# assumes x is a time vector and returns a linearly increasing function with the same shape as x
def ramp_func(x0, x1, times):
    x = np.zeros_like(times)
    x[0] = x0
    x[-1] = x1
    for i in range(1, len(times)-1):
        x[i] = x0 + (x1 - x0) * (times[i] - times[0]) / (times[-1] - times[0])
    return x

def interpolate_func(x, x_points, y_points):
    f_interp = interpolate.interp1d(x_points, y_points, kind='linear', fill_value='extrapolate')
    return f_interp(x)

# Random walk function that generates a random walk signal with the same shape as input
# x[i+1] = x[i] \pm 1, then scaled w.r.t. max_amp with saturation at max_amp
def random_walk(x, seed):
    start_amp = 30
    walk_amp = 2
    max_amp = 50
    rng = np.random.default_rng(seed)
    sz = np.size(x)
    x_out = np.zeros(sz)
    x_step = np.round(rng.uniform(-walk_amp, walk_amp, sz))
    
    for i in range(len(x)-1):
        x_out[i+1] = x_out[i] + 2.0*np.round(np.random.uniform())-1
        x_out[i+1] = np.sign(x_out[i+1]) * np.min([np.abs(x_out[i+1]), max_amp])
    
    return x_out 

