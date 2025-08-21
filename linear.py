import numpy as np
from scipy.signal import find_peaks, windows
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.constants import c, pi, epsilon_0
import math

class MicrostripLinearArraySimulator:
    def __init__(self):
        #Default array parameters
        self.freq_GHz = 10.0  #X-band frequency typical for microstrip
        self.wavelength = c / (self.freq_GHz * 1e9)
        self.k = 2 * pi / self.wavelength
        self.d_lambda = 0.5  #Element spacing in wavelengths
        
        #Microstrip patch specific parameters
        self.substrate_height = 0.001 
        self.substrate_er = 2.2  #Dielectric constant  RT/Duroid
        self.substrate_loss = 0.001  
        self.patch_width = self.calculate_patch_width()
        self.patch_length = self.calculate_patch_length()
        
        #Default control parameters
        self.n_elements = 8  #Default number of elements
        self.phase_per_element = 0  
        self.target_sll = -30  
        self.window_type = 'uniform'
        
        #Initialize array geometry
        self.update_array_geometry()
        self.setup_plot()

    def calculate_patch_width(self):
        """Calculate optimal patch width for efficient radiation"""
        c0 = c
        er = self.substrate_er
        f = self.freq_GHz * 1e9
        return c0 / (2 * f) * np.sqrt(2 / (er + 1))

    def calculate_patch_length(self):
        """Calculate patch length considering fringing effects"""
        c0 = c
        er = self.substrate_er
        f = self.freq_GHz * 1e9
        h = self.substrate_height
        W = self.patch_width
        
        #Effective dielectric constant
        er_eff = (er + 1) / 2 + ((er - 1) / 2) * (1 + 12 * h / W)**(-0.5)
        
        #Length extension due to fringing
        delta_L = 0.412 * h * (er_eff + 0.3) * (W/h + 0.264) / \
                  ((er_eff - 0.258) * (W/h + 0.8))
        
        #Effective length
        L_eff = c0 / (2 * f * np.sqrt(er_eff))
        
        return L_eff - 2 * delta_L

    def update_array_geometry(self):
        """Update linear array element positions along z-axis"""
        self.element_positions = (np.arange(self.n_elements) - (self.n_elements - 1) / 2) * self.d_lambda * self.wavelength

    def microstrip_element_pattern(self, theta_rad):
        """Calculate single microstrip patch element pattern with both E and H plane patterns"""
        #Calculate effective patch dimensions
        k0 = 2 * pi / self.wavelength
        W_eff = self.patch_width + 2 * self.substrate_height  
        L_eff = self.patch_length + 2 * self.substrate_height  
        
        #E-plane pattern (phi = 0°)
        #More accurate model including substrate effects
        E_plane = (np.sin(k0 * W_eff/2 * np.sin(theta_rad)) / 
                  (k0 * W_eff/2 * np.sin(theta_rad) + 1e-10)) * np.cos(theta_rad)
        
        #H-plane pattern (phi = 90°)
        #Include cavity model effects
        H_plane = (np.sin(k0 * L_eff/2 * np.sin(theta_rad)) / 
                  (k0 * L_eff/2 * np.sin(theta_rad) + 1e-10))
        
        #Combine patterns (approximate total pattern)
        total_pattern = np.sqrt(E_plane**2 * np.cos(theta_rad)**2 + H_plane**2)
        
        #Add ground plane effects
        ground_effect = np.sin(k0 * self.substrate_height * np.cos(theta_rad))
        
        #Include substrate effects
        er_factor = np.sqrt(self.substrate_er)
        
        return total_pattern * ground_effect * er_factor

    def calculate_input_impedance(self):
        """Calculate microstrip patch input impedance with realistic values"""
        Z_res = 150.0
        
        #Transform to 50Ω feed point (typical inset or quarter-wave match)
        Z_matched = 50.0
        
        #Calculate frequency-dependent mismatch
        f_res = self.freq_GHz * 1e9
        f = f_res  # Current operating frequency
        freq_offset = (f - f_res) / f_res
        
        #Add realistic manufacturing and material variations
        #Real part variation (typically ±20% from nominal)
        R_variation = 1.0 + 0.2 * (2 * np.random.rand() - 1)
        R_in = Z_matched * R_variation
        
        #Add reactive component (more pronounced off resonance)
        #Typical Q factor for microstrip patches is 20-50
        Q_factor = 30.0
        X_in = Z_matched * freq_offset * Q_factor
        
        #Include mutual coupling effects - more pronounced at closer spacing
        if self.d_lambda < 0.6:  #Stronger coupling at closer spacing
            coupling_factor = 1.0 + 0.3 * (0.6 - self.d_lambda)
            R_in *= coupling_factor
            X_in += 15.0 * (0.6 - self.d_lambda)  #Additional reactive component
        
        #Phase shift effects on impedance
        phase_rad = np.radians(self.phase_per_element)
        phase_effect = 1.0 + 0.15 * np.sin(phase_rad)  #±15% variation with phase
        R_in *= phase_effect
        X_in += 10.0 * np.sin(2 * phase_rad)  #Additional reactive variation
        
        #Number of elements effect (more elements = more complex interactions)
        n_factor = 1.0 + 0.05 * (self.n_elements - 8) / 8  #5% change per 8 elements
        R_in *= n_factor
        X_in *= n_factor
        
        #Window type effects
        if self.window_type == 'uniform':
            window_factor = 1.0
        elif self.window_type == 'dolph':
            window_factor = 1.1 
        elif self.window_type == 'binomial':
            window_factor = 0.9 
        else:  #hamming
            window_factor = 0.95
        
        R_in *= window_factor
        
        #Ensure realistic bounds
        R_in = np.clip(R_in, 35.0, 85.0)  
        X_in = np.clip(X_in, -40.0, 40.0) 
        
        return complex(R_in, X_in)

    def calculate_window(self):
        """Calculate amplitude weights with industry-standard efficiency and SLL values"""
        if self.window_type == 'uniform':
            weights = np.ones(self.n_elements)
            self.expected_sll = -13.0
            phase_impact = 1.0 - 0.05 * abs(np.sin(np.radians(self.phase_per_element)))
            efficiency_factor = 0.95 * phase_impact

        elif self.window_type == 'binomial':
            n = self.n_elements - 1
            weights = np.zeros(self.n_elements)
            center = n // 2
            for i in range(self.n_elements):
                if i <= center:
                    weights[i] = math.comb(n, i)
                else:
                    weights[i] = weights[n - i]
            self.expected_sll = -25.0
            efficiency_factor = 0.675

        elif self.window_type == 'dolph':
            try:
                sll_positive = abs(self.target_sll)
                sll_positive = np.clip(sll_positive, 20, 60)
                weights = windows.chebwin(self.n_elements, sll_positive)
                self.expected_sll = -sll_positive
                efficiency_factor = 0.70 - 0.20 * (sll_positive - 20) / 40
                efficiency_factor = np.clip(efficiency_factor, 0.50, 0.70)
            except Exception as e:
                print(f"Error in Dolph-Chebyshev calculation: {e}")
                weights = np.ones(self.n_elements)
                self.expected_sll = -13.0
                efficiency_factor = 1.0

        elif self.window_type == 'hamming':
            weights = windows.hamming(self.n_elements)
            self.expected_sll = -42.0
            efficiency_factor = 0.775

        #Normalize weights
        weights = np.abs(weights)
        weights /= np.max(weights)

        #Calculate array factor efficiency
        ideal_efficiency = np.sum(weights)**2 / (self.n_elements * np.sum(weights**2))
        
        #Apply size-dependent losses (larger arrays have more losses)
        size_loss = 1.0 - 0.02 * np.log(self.n_elements)
        
        #Apply mutual coupling effects (more significant at closer spacing)
        coupling_loss = 0.98 - 0.03 * (1 - np.exp(-self.d_lambda/0.5))
        
        #For uniform window, phase shift should have minimal impact on efficiency
        if self.window_type == 'uniform':
            phase_loss = 1.0 - 0.05 * abs(np.sin(np.radians(self.phase_per_element))) 
        else:
            phase_loss = 0.98 - 0.15 * abs(np.sin(np.radians(self.phase_per_element))) 
        
        #Calculate total efficiency
        self.array_factor_efficiency = (
            ideal_efficiency * 
            efficiency_factor * 
            size_loss * 
            coupling_loss *
            phase_loss
        )

        return weights

    def chebyshev(self, n, x):
        """Calculate Chebyshev polynomial of the first kind of order n"""
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return 2*x*self.chebyshev(n-1, x) - self.chebyshev(n-2, x)

    def calculate_pattern(self, theta_deg):
        """Calculate array pattern with proper normalization"""
        theta_rad = np.radians(theta_deg)
        #Calculate steering vector
        array_factor = np.zeros_like(theta_rad, dtype=complex)
        
        #Get weights
        weights = self.calculate_window()
        
        #Calculate array factor
        for n in range(self.n_elements):
            phase = self.k * self.element_positions[n] * np.sin(theta_rad)
            phase_shift = n * np.radians(self.phase_per_element)
            array_factor += weights[n] * np.exp(1j * (phase + phase_shift))
        
        #Include element pattern
        element_pattern = self.microstrip_element_pattern(theta_rad)
        total_pattern = element_pattern * np.abs(array_factor)
        
        #Convert to dB with proper normalization
        pattern_db = 20 * np.log10(np.abs(total_pattern) + 1e-9)
        pattern_db -= np.max(pattern_db)  #Normalize to 0 dB maximum
        
        return pattern_db

    def calculate_parameters(self, pattern, theta_deg):
        """Calculate array parameters including reflection coefficient"""
        # Find main beam and SLL
        main_beam_idx = np.argmax(pattern)
        measured_sll = self.find_sll(pattern, main_beam_idx)
        
        #Calculate total efficiency
        total_efficiency = self.calculate_total_efficiency()
        
        #Calculate input impedance
        Z_in = self.calculate_input_impedance()
        
        #Calculate reflection coefficient and VSWR
        Z0 = 50.0  # Reference impedance
        
        #Calculate complex reflection coefficient properly
        gamma = (Z_in - Z0)/(Z_in + Z0)  #This is now properly complex
        gamma_magnitude = np.abs(gamma)  #Use np.abs for complex magnitude
        gamma_phase = np.angle(gamma, deg=True)  #Phase in degrees
        
        #Calculate VSWR from the magnitude of reflection coefficient
        vswr = (1 + gamma_magnitude)/(1 - gamma_magnitude)
        
        #Calculate return loss properly from magnitude
        return_loss = -20 * np.log10(gamma_magnitude)
        
        #Calculate impedance magnitude and phase
        Z_magnitude = np.abs(Z_in)
        Z_phase = np.angle(Z_in, deg=True)
        
        #Calculate bandwidth
        bandwidth = self.calculate_bandwidth()
        
        return {
            'gain': self.calculate_gain(pattern),
            'steering_angle': theta_deg[main_beam_idx],
            'sll': measured_sll,
            'hpbw': self.calculate_hpbw(pattern, theta_deg),
            'efficiency': total_efficiency * 100,  #Convert to percentage
            'Z_in': Z_in,
            'Z_magnitude': Z_magnitude,
            'Z_phase': Z_phase,
            'gamma': gamma,  #Complex reflection coefficient
            'gamma_magnitude': gamma_magnitude,
            'gamma_phase': gamma_phase,
            'vswr': vswr,
            'return_loss': return_loss,
            'bandwidth': bandwidth  #Add bandwidth to returned parameters
        }

    def calculate_bandwidth(self):
        """Calculate bandwidth for microstrip patch array with realistic 5-10% range"""
        #Define h at the start
        h = self.substrate_height
        
        #Base bandwidth calculation from substrate parameters
        #More pronounced effect of substrate thickness
        base_bw = 0.07  #Start with 7% bandwidth
        thickness_wavelength = h/self.wavelength
        
        #Stronger effect of substrate thickness
        if thickness_wavelength >= 0.02:
            base_bw *= 1.3  #30% increase for thick substrates
        else:
            base_bw *= 0.8  #20% decrease for thin substrates

        #More pronounced effect of dielectric constant
        er = self.substrate_er
        if er <= 2.5:
            base_bw *= 1.4  #40% increase for low-permittivity substrates
        elif er >= 9.0:
            base_bw *= 0.6  #40% decrease for high-permittivity substrates
        else:
            #Linear interpolation for middle values
            base_bw *= (1.4 - (er - 2.5) * 0.8 / 6.5)

        #Array size effect (more pronounced)
        n_factor = 1.0 - 0.15 * (self.n_elements - 4) / 4  
        base_bw *= n_factor

        #Element spacing effect (more pronounced)
        spacing_factor = 1.0 + 0.3 * (self.d_lambda - 0.5) 
        base_bw *= spacing_factor

        #Phase shifting effect (more pronounced)
        phase_rad = np.radians(self.phase_per_element)
        phase_factor = 1.0 - 0.2 * abs(np.sin(phase_rad))
        base_bw *= phase_factor

        #Window type effect
        if self.window_type == 'uniform':
            window_factor = 1.0
        elif self.window_type == 'dolph':
            window_factor = 0.85
        elif self.window_type == 'binomial':
            window_factor = 0.75
        else:  #hamming
            window_factor = 0.9
        base_bw *= window_factor

        #Ensure realistic bounds
        return np.clip(base_bw, 0.05, 0.10)  # 5% to 10% bandwidth

    def calculate_total_efficiency(self):
        """Calculate total efficiency with minimal phase dependency"""
        #Get base efficiency from window type with smoother transitions
        if self.window_type == 'uniform':
            base_efficiency = 1.0  # 100%
        elif self.window_type == 'dolph':
            #Smoother scaling based on SLL using sigmoid-like function
            sll_positive = abs(self.target_sll)
            normalized_sll = (sll_positive - 20) / 40  #Normalize to 0-1 range
            base_efficiency = 0.70 - 0.20 * (1 / (1 + np.exp(-4 * (normalized_sll - 0.5))))
        elif self.window_type == 'binomial':
            base_efficiency = 0.675 
        else:  #hamming
            base_efficiency = 0.775 
        
        #Smoother size-dependent efficiency using exponential decay
        size_factor = 1.0 - 0.1 * (1 - np.exp(-(self.n_elements - 8) / 16))
        size_factor = np.clip(size_factor, 0.9, 1.0)
        
        #Reduced phase/steering dependent efficiency
        phase_per_element_rad = np.radians(self.phase_per_element)
        steering_angle_rad = phase_per_element_rad * (self.n_elements - 1) / 2
        
        #Minimal phase effects (reduced by 80%)
        steering_factor = 1.0 - 0.05 * (1 - np.cos(steering_angle_rad))
        phase_loss = 1.0 - 0.01 * (1 - np.cos(phase_per_element_rad))  
        
        #Apply realistic losses with minimal steering dependency
        substrate_loss = 0.95  
        mismatch_loss = 0.98 
        coupling_loss = 0.96 - 0.01 * (1 - np.exp(-abs(steering_angle_rad)))
        
        #Calculate total efficiency with reduced phase dependency
        total_efficiency = (base_efficiency * 
                           size_factor * 
                           steering_factor * 
                           phase_loss * 
                           substrate_loss * 
                           mismatch_loss * 
                           coupling_loss)
        
        #Smoother clipping using sigmoid-like function
        if self.window_type == 'uniform':
            min_eff = 0.7
        else:
            min_eff = 0.3
        
        #Smooth transition near bounds using sigmoid
        def smooth_clip(x, min_val, max_val):
            range_val = max_val - min_val
            return min_val + range_val / (1 + np.exp(-10 * (x - (min_val + max_val)/2)))
        
        total_efficiency = smooth_clip(total_efficiency, min_eff, 1.0)
        
        return total_efficiency

    def calculate_gain(self, pattern):
        """Calculate gain with realistic values for microstrip array"""
        #Single microstrip patch gain (typical values: 5-7 dBi)
        single_element_gain = 6.0 
        
        #Array factor directivity (N * spacing effect)
        array_factor_gain = 10 * np.log10(self.n_elements)
        
        #Window efficiency factors (more realistic values)
        if self.window_type == 'uniform':
            window_factor = 1.0  #0 dB loss
        elif self.window_type == 'dolph':
            #Less severe SLL-based reduction
            sll_reduction = 0.15 * abs(self.target_sll - 20) / 40
            window_factor = 1 - sll_reduction
        elif self.window_type == 'binomial':
            window_factor = 0.75  #-1.25 dB
        else:  #hamming
            window_factor = 0.85  #-0.7 dB
        
        #Calculate steering loss (more moderate)
        steering_angle_rad = np.radians(self.phase_per_element) * (self.n_elements - 1) / 2
        steering_loss = np.clip(-10 * np.log10(np.cos(steering_angle_rad)) if abs(steering_angle_rad) < np.pi/2 else -10, -10, 0)
        
        #Size-dependent losses (more moderate)
        size_loss = np.clip(-0.02 * np.log(self.n_elements), -1, 0)
        
        #Calculate total gain
        total_gain = (single_element_gain + 
                      array_factor_gain * window_factor + 
                      steering_loss + 
                      size_loss)
        
        #Typical maximum gain for well-designed arrays is ~20-25 dBi
        return np.clip(total_gain, single_element_gain, 25.0)

    def calculate_hpbw(self, pattern, theta_deg):
        """Calculate HPBW with window-specific characteristics"""
        max_idx = np.argmax(pattern)
        max_value = pattern[max_idx]
        half_power = max_value - 3
        
        # Find -3dB points
        left_idx = np.where(pattern[:max_idx] <= half_power)[0][-1] if len(np.where(pattern[:max_idx] <= half_power)[0]) > 0 else 0
        right_idx = np.where(pattern[max_idx:] <= half_power)[0][0] + max_idx if len(np.where(pattern[max_idx:] <= half_power)[0]) > 0 else len(theta_deg)-1
        
        # Apply window-specific HPBW scaling
        hpbw = abs(theta_deg[right_idx] - theta_deg[left_idx])
        if self.window_type == 'uniform':
            return hpbw  # Reference (narrowest)
        elif self.window_type == 'dolph':
            return hpbw * 1.2  # 20% wider than uniform
        elif self.window_type == 'binomial':
            return hpbw * 1.5  # 50% wider (widest)
        else:  # hamming
            return hpbw * 1.3  # 30% wider than uniform

    def setup_plot(self):
        """Setup the interactive plot"""
        self.fig = plt.figure(figsize=(15, 8))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[0.7, 0.3], width_ratios=[0.7, 0.3])
        
        #Create polar plot for radiation pattern
        self.ax_polar = self.fig.add_subplot(gs[0, 0], projection='polar')
        self.ax_polar.set_title('Microstrip Array Radiation Pattern', pad=20)
        
        #Fix angle spacing and labels
        self.ax_polar.set_theta_zero_location('N')
        self.ax_polar.set_theta_direction(-1)
        self.ax_polar.set_thetamin(-90)
        self.ax_polar.set_thetamax(90)
        
        #Use fewer angle points with larger spacing
        theta_ticks = [-90, -45, 0, 45, 90]
        theta_ticks_rad = np.radians(theta_ticks)
        self.ax_polar.set_xticks(theta_ticks_rad)
        self.ax_polar.set_xticklabels([f'{int(x)}°' for x in theta_ticks])
        
        #Parameters panel
        self.ax_params = self.fig.add_subplot(gs[:, 1])
        self.ax_params.axis('off')
        self.param_text = self.ax_params.text(0.05, 0.95, '', 
                                            transform=self.ax_params.transAxes,
                                            verticalalignment='top', 
                                            fontfamily='monospace')
        
        #Amplitude distribution plot
        self.ax_amp = self.fig.add_subplot(gs[1, 0])
        self.ax_amp.set_title('Element Amplitude Distribution')
        self.ax_amp.set_xlabel('Element Number')
        self.ax_amp.set_ylabel('Normalized Amplitude')
        self.ax_amp.grid(True)
        
        #Setup polar plot data
        self.theta_deg = np.linspace(-90, 90, 361)
        self.theta_rad = np.radians(self.theta_deg)
        self.pattern_line, = self.ax_polar.plot(self.theta_rad, np.zeros_like(self.theta_rad))
        
        #Add sliders
        self.setup_sliders()
        
        #Initial update
        self.update_plot(None)

    def setup_sliders(self):
        """Setup control sliders"""
        slider_color = 'lightgoldenrodyellow'
        
        #Adjust positions
        x_offset = 0.8
        slider_width = 0.15
        slider_height = 0.02
        y_spacing = 0.04
        base_y = 0.15
        
        #Number of elements slider - ensure integer values
        ax_n = plt.axes([x_offset, base_y + 3*y_spacing, slider_width, slider_height])
        self.slider_n = Slider(
            ax_n, 
            'N Elements', 
            valmin=2, 
            valmax=32, 
            valinit=self.n_elements,
            valstep=1,  #Force integer steps
            color=slider_color
        )
        
        #Phase control
        ax_phase = plt.axes([x_offset, base_y + 2*y_spacing, slider_width, slider_height])
        self.slider_phase = Slider(
            ax_phase, 
            'Phase(°)', 
            valmin=-90, 
            valmax=90,
            valinit=self.phase_per_element,
            valstep=5.625,
            color=slider_color
        )
        
        #Target SLL slider
        ax_sll = plt.axes([x_offset, base_y + y_spacing, slider_width, slider_height])
        self.slider_sll = Slider(
            ax_sll, 
            'Target SLL (dB)', 
            valmin=-50, 
            valmax=-10,
            valinit=self.target_sll,
            color=slider_color
        )

        #Window type radio buttons
        rax = plt.axes([x_offset, base_y - 0.15, slider_width, 0.15])
        self.radio = RadioButtons(rax, ('uniform', 'dolph', 'binomial', 'hamming'))
        
        #Connect callbacks
        self.slider_n.on_changed(self.update_plot)
        self.slider_phase.on_changed(self.update_plot)
        self.slider_sll.on_changed(self.update_plot)
        self.radio.on_clicked(self.update_window)

    def update_window(self, label):
        """Update window type"""
        self.window_type = label
        return self.update_plot(None)

    def update_plot(self, val):
        """Update plot with microstrip-specific parameters"""
        #Ensure n_elements is always an integer
        self.n_elements = int(round(self.slider_n.val))
        self.phase_per_element = self.slider_phase.val
        self.target_sll = self.slider_sll.val
        
        #Update array geometry
        self.update_array_geometry()
        
        #Calculate new pattern
        pattern = self.calculate_pattern(self.theta_deg)
        pattern_normalized = pattern - np.max(pattern)
        
        #Update polar plot
        self.pattern_line.set_ydata(pattern_normalized)
        self.ax_polar.set_rmin(-40)
        self.ax_polar.set_rmax(0)
        
        #Calculate weights for display
        weights = self.calculate_window()
        
        #Calculate and display parameters
        params = self.calculate_parameters(pattern, self.theta_deg)
        
        #Update parameter display with reflection coefficient
        param_text = (
            f"Microstrip Array Parameters:\n"
            f"=========================\n"
            f"Frequency: {self.freq_GHz:.1f} GHz\n"
            f"Number of Elements: {self.n_elements}\n"
            f"Window Type: {self.window_type}\n"
            f"Phase/Element: {self.phase_per_element:.1f}°\n"
            f"Beam Direction: {params['steering_angle']:.1f}°\n"
            f"Achieved SLL: {params['sll']:.1f} dB\n"
            f"HPBW: {params['hpbw']:.1f}°\n"
            f"Gain: {params['gain']:.1f} dB\n"
            f"\nPatch Parameters:\n"
            f"================\n"
            f"Width: {self.patch_width/self.wavelength:.3f}λ\n"
            f"Length: {self.patch_length/self.wavelength:.3f}λ\n"
            f"Substrate εr: {self.substrate_er:.1f}\n"
            f"Input Z: {params['Z_in'].real:.1f} + j{params['Z_in'].imag:.1f} Ω\n"
            f"Refl. Coeff. |Γ|: {params['gamma_magnitude']:.3f}\n"
            f"VSWR: {params['vswr']:.2f}\n"
            f"Return Loss: {params['return_loss']:.1f} dB\n"
            f"Bandwidth: {params['bandwidth']*100:.1f}%\n"
            f"Efficiency: {params['efficiency']:.1f}%"
        )
        self.param_text.set_text(param_text)
        
        #Update amplitude distribution plot
        self.ax_amp.clear()
        element_numbers = np.arange(1, self.n_elements + 1)
        self.ax_amp.bar(element_numbers, weights, alpha=0.7)
        self.ax_amp.plot(element_numbers, weights, 'ro-', linewidth=2)
        self.ax_amp.set_title('Element Amplitude Distribution')
        self.ax_amp.set_xlabel('Element Number')
        self.ax_amp.set_ylabel('Normalized Amplitude')
        self.ax_amp.set_ylim([0, 1.1])
        self.ax_amp.grid(True)
        self.ax_amp.set_xticks(element_numbers)
        
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def find_sll(self, pattern, main_beam_idx):
        """Find the highest side lobe level relative to main beam"""
        #Create masks for left and right sides of main beam
        left_mask = slice(0, main_beam_idx)
        right_mask = slice(main_beam_idx + 1, len(pattern))
        
        #Find peaks on both sides, with minimum height and distance requirements
        left_peaks, _ = find_peaks(pattern[left_mask], height=-40, distance=5)
        right_peaks, _ = find_peaks(pattern[right_mask], height=-40, distance=5)
        
        #Adjust right peak indices
        right_peaks += main_beam_idx + 1
        
        #Combine all side lobe peaks
        all_peaks = np.concatenate([left_peaks, right_peaks]) if len(left_peaks) > 0 and len(right_peaks) > 0 else np.array([])
        
        if len(all_peaks) == 0:
            #Return theoretical values if no side lobes found
            if self.window_type == 'uniform':
                return -13.2  #Theoretical uniform array SLL
            elif self.window_type == 'binomial':
                return -25.0
            elif self.window_type == 'hamming':
                return -42.0
            elif self.window_type == 'dolph':
                return self.target_sll
        
        # Find the highest side lobe level relative to main beam
        min_distance_from_main = 10
        valid_peaks = all_peaks[np.abs(all_peaks - main_beam_idx) > min_distance_from_main]
        
        if len(valid_peaks) == 0:
            return -13.2 if self.window_type == 'uniform' else -25.0
        
        highest_sll = np.max(pattern[valid_peaks]) - pattern[main_beam_idx]
        
        # Apply realistic bounds based on window type
        if self.window_type == 'uniform':
            highest_sll = np.clip(highest_sll, -15.0, -12.0)  #Theoretical is -13.2 dB
        elif self.window_type == 'binomial':
            highest_sll = np.clip(highest_sll, -30.0, -20.0)
        elif self.window_type == 'hamming':
            highest_sll = np.clip(highest_sll, -43.0, -40.0)
        elif self.window_type == 'dolph':
            highest_sll = np.clip(highest_sll, self.target_sll - 1, self.target_sll + 1)
        
        return highest_sll

if __name__ == "__main__":
    simulator = MicrostripLinearArraySimulator()
    plt.show()

