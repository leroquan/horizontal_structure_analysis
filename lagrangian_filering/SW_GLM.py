import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.fft import rfft2, irfft2
import scipy.io as spio
from scipy.special import sici
import time
from tqdm import tqdm
import os
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator
import matplotlib
from pathlib import Path
import pickle
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

"""
The module 'SW_GLM' contains the SWSolver class, which solves the (modified) shallow water equations in a 2D periodic domain, 
alongside an online generalised Lagrangian mean. 
"""

class SWSolver:
    """ SWSolver solves the shallow water equations (with options for modified shallow water or linearised equations) alongside Lagrangian mean 
    equations to find the weighted Lagrangian mean at user defined times in the simulation, using one of three different strategies.
    """

    def __init__(self,strat=3,Re=None, Re_hyp=4e13, Ro=0.1, Fr = 0.5, save_inst=False,
                 save_inst_dt = 0.05, ny_inst = None, Nx=0, Ny=0, dt=2.5e-4, T = 1, Nint=1, dtslow = 1, t_LM_start=0,solve_LM=True,MSW=False,linear=False,
                 movie_type=None,movie_fname=f'mymovie',movie_dir='./', movie_freq=10,
                 save_solver=False,save_dir='./'):
        
        """ Initialise the physical parameters of the problem, the numerical parameters, the solver options, and the real and the physical and spectral grids""

        Args:
            strat (int, optional): Lagrangian mean solver strategy, takes values 1, 2 or 3. Defaults to 3.
            Re (float, optional): Reynolds number for shallow water equations. Defaults to None.
            Re_hyp (float, optional): Hyperviscous Reynolds number for shallow water equations. Defaults to 4e13.
            Ro (float, optional): Rossby number. Defaults to 0.1.
            Fr (float, optional): Froude number. Defaults to 0.5.
            save_inst (bool, optional): If true, forms a dataset of instantaneous variables at interval save_inst_dt
            save_inst_dt (float, optional): Saving time interval for instantaneous fields. Defaults to 0.05.
            ny_inst (float, optional): y gridcell to save instantaneous fields at. Defaults to None.
            Nx (int, optional): Grid size, with pre-set initial conditions can take value 128 or 256. Defaults to 128.
            dt (float, optional): Timestep. Defaults to 2.5e-4.
            Nint (int, optional): Number of Lagrangian mean intervals. Defaults to 1
            T (float, optional): Interval time for Lagrangian mean. Defaults to 1.
            Nint (int, optional): Number of intervals to run Lagrangian mean equations over. Defaults to 1. 
            dtslow (float, optional): Spacing between Lagrangian means. Defaults to 1.
            t_LM_start (float, optional): Time at which to start first Lagrangian mean equation. Defaults to 0.
            solve_LM (bool, optional): Solve Lagrangian mean equations alongside shallow water when true. Defaults to True.
            MSW (bool, optional): Solve modified shallow water rather than shallow water. Defaults to False.
            linear (bool, optional): Solve linear equations (the same for shallow water and modified shallow water). Defaults to False.
            movie_type (str, optional): Denotes type of movie to make. Options are 'vorticity','pv','u velocity','v velocity', 'vorticity means', 'jacobian', Defaults to None.
            movie_fname (str, optional): Movie filename. Defaults to 'mymovie'.
            movie_dir (str, optional): save location for movies. Defaults to './'
            movie_freq (int, optional): Multiple of timestep at which to plot movie frames. Defaults to 10.
            save_solver (bool, optional): If True, pickles entire solver object at the end. Defaults to False
            save_dir (str, optional): Location to save pickled solver object. Defaults to './'
        """
        if Nx % 2 != 0:
            raise ValueError('Nx should be even')
        
        # Initialise attributes
        self.Nx = Nx
        self.Ny = Ny
        self.Ro = Ro
        self.Fr = Fr
        self.strat = strat
        self.solve_LM = solve_LM
        self.MSW = MSW
        self.linear = linear

        # Set up time intervals
        self.dtslow = dtslow
        self.Nint = Nint
        self.t = 0

        # Adjust timestep dt so that dtslow is a multiple of dt
        self.dt = self.dtslow/int(self.dtslow/dt)
        print(f'Timestep dt = {self.dt}')

        # Adjust interval length T so that T/2 is a multiple of dt
        self.T = int((T/2)/self.dt)*self.dt*2
        print(f'Interval time T = {self.T}')

        # Set end time
        self.Ttotal = t_LM_start+self.T + (self.Nint - 1)*self.dtslow
        print(f'Total time is {self.Ttotal}')

        # Get number of timesteps
        self.Nttotal = int(self.Ttotal/self.dt)
        self.Ntint = int(self.T/self.dt)
        self.Ntinthf = int(self.Ntint/2)
        self.Ntgap = int(self.dtslow/self.dt)
        self.Nt_save_inst=int(save_inst_dt/self.dt)

        # Set interval start times
        self.LM_start_step = int(t_LM_start/self.dt)
        
        # Update t_LM_start to be a multiple of timesteps:
        t_LM_start = self.LM_start_step*self.dt
        print(f'Lagrangian interval start time updated to {t_LM_start}')
        self.interval_start = np.linspace(t_LM_start,self.dtslow*self.Nint+t_LM_start, self.Nint,endpoint=False)

        # Set flag variable that determines the state of each interval: 
        # 0: Interval not being solved
        # 1: First half being solved
        # 2: Second half being solved
        self.flag = np.zeros((self.Nint))

        # Set movie parameters
        self.movie_type = movie_type
        self.movie_fname = movie_fname
        self.movie_freq = movie_freq
        self.movie_dir= movie_dir
        if (self.Nint > 1) and (movie_type != 'vorticity') and (movie_type is not None):
            print('Just run for one interval to get movies of Lagrangian variables. Not saving a movie.')
            self.movie_type = None

        self.save_solver = save_solver
        self.save_dir = save_dir
        self.save_inst = save_inst
        if self.save_inst:
            if ny_inst is None:
                self.ny_inst = int(self.Ny/2)
            else:
                self.ny_inst = ny_inst
        
        ## ========================== Spatial Variables + Wavenumbers ==========================
        
        # grid spacing in real space
        self.dx = 2*np.pi/self.Nx
        self.dy = 2*np.pi/self.Ny

        # grid points in real space
        self.x = (np.arange(0,self.Nx))*self.dx
        self.y = (np.arange(0,self.Ny))*self.dy

        # meshgrid of the points
        self.xx, self.yy = np.meshgrid(self.x,self.y)

        # padding xx and yy only for interpolation (according to periodic BC)
        self.x_padd = (np.arange(0,self.Nx+1))*self.dx
        self.y_padd = (np.arange(0,self.Ny+1))*self.dy
        self.xx_padd, self.yy_padd = np.meshgrid(self.x_padd,self.y_padd)

        # In Fourier Space
        # The x axis is shorter as the rfft is performed in this dimension
        self.Kx = np.zeros(int(self.Nx/2+1)) 
        self.Ky = np.zeros(self.Ny)

        self.Kx = np.arange(int(self.Nx/2+1)) 
        self.Ky[:int(self.Ny/2+1)] = np.arange(int(self.Ny/2+1))
        self.Ky[int(self.Ny/2+1):] = np.arange(-int(self.Ny/2)+1,0) 
        self.Kxx, self.Kyy = np.meshgrid(self.Kx,self.Ky)

        #  k2 is used for dissipation term i.e. -nu*k^2 
        #  (where we dont have issue with k2 =0)
        #  k2poisson is for solving poisson equation, so can't have zeros
        self.k2 = self.Kxx**2+self.Kyy**2
        self.k2poisson = self.Kxx**2+self.Kyy**2
        self.k2poisson[0,0] = 1

        #---------------------------------------------------------------
        #  Calculate the viscosity coefficients for the shallow water equations
        if (Re_hyp is None) and (Re is None):
            self.Cc = 1
            print('Warning: no viscosity')
        elif (Re_hyp is None):
            self.Cc = np.exp(-self.dt/Re*self.k2)
        elif (Re is None):
            self.Cc = np.exp(-self.dt/Re_hyp*self.k2**4)
        else: 
            self.Cc = np.exp(-self.dt/Re_hyp*self.k2**4 -self.dt/Re*self.k2)

        #  De-aliasing mask: forces the nonlinear terms for kx,ky>2/3 to be zero
        #  depending on the problem and the type of dissipation can be relaxed
        self.L = np.ones(np.shape(self.k2poisson))
        for i in range(int(self.Nx/2)+1):
            for j in range(self.Ny):
                if abs(self.Kxx[j, i]) > max(self.Kx)*2./3.:
                    self.L[j, i] = 0
                elif abs(self.Kyy[j, i]) > max(self.Ky)*2./3.:
                    self.L[j, i] = 0

    def set_SW_IC(self,IC_file='./uvr_2Dturbulence_256.mat', A = 0,kw=1,lw=0):

        """ Set initial conditions of Shallow Water equations

        Args:
            IC_file (str, optional): .mat file containing u and v components of initial flow. Defaults to 'uvr_2Dturbulence_256.mat'.
            A (float or list(float), optional): wave component vorticity amplitude. Use list for multiple wave components. Defaults to 0.
            kw (int or list(int), optional): wave component x-wavenumber. Defaults to 1.
            lw (int or list(int), optional): wave component y-wavenumber. Defaults to 0.

        """
        # =================== initialising the real field quantities ===================
        p = Path(__file__).with_name(IC_file)
        ICfilepath = p.absolute()
        if os.path.isfile(ICfilepath):
            ur_in = spio.loadmat(ICfilepath)['ur']
            vr_in = spio.loadmat(ICfilepath)['vr']
            if self.Nx == 256:
                self.ur = ur_in
                self.vr = vr_in
            elif self.Nx == 128:
                self.ur = ur_in[0:-1:2,0:-1:2]
                self.vr = vr_in[0:-1:2,0:-1:2]
            else:
                x_in_pad = (np.arange(0,257))*2*np.pi/256
                y_in_pad = (np.arange(0,257))*2*np.pi/256
                ur_pad = self._pad(ur_in)
                vr_pad = self._pad(vr_in)
                interp_ur = RectBivariateSpline(y_in_pad, x_in_pad, ur_pad, kx=3, ky=3)
                interp_vr = RectBivariateSpline(y_in_pad, x_in_pad, vr_pad, kx=3, ky=3)
                ur_interp_ravel = interp_ur.ev(np.ravel(self.yy), np.ravel(self.xx))
                vr_interp_ravel = interp_vr.ev(np.ravel(self.yy), np.ravel(self.xx))
                self.ur = np.reshape(ur_interp_ravel, (self.xx.shape[0], self.xx.shape[1]))
                self.vr = np.reshape(vr_interp_ravel, (self.xx.shape[0], self.xx.shape[1]))
     
        else:
            print(f'{IC_file} does not exist and no other initialisation given. Using zero non-wave IC.')
            self.ur = np.zeros_like(self.xx)
            self.vr = np.zeros_like(self.xx)
        
        self.uk = rfft2(self.ur)
        self.vk = rfft2(self.vr)

        # Define vorticity
        self.zk = 1j*self.Kxx*self.vk-1j*self.Kyy*self.uk
        self.zk[0,0] = 0 # Make sure that the zero wavenumber mode is zero, as we get inertial oscillations otherwise
        self.zk[int(self.Ny/2),:] = 0 # Set Nyquist frequency to zero as no conjugate symmetry for even N
        self.zk[:,int(self.Nx/2)] = 0

        # Now redefine all fields from vorticity to make sure they are consistent with geostrophic balance
        
        self.hk = -(self.Fr**2/self.Ro)*self.zk/self.k2poisson
        self.uk = -1j*self.Kyy*self.Ro/self.Fr**2*self.hk
        self.vk = 1j*self.Kxx*self.Ro/self.Fr**2*self.hk

        self.ur = irfft2(self.uk)
        self.vr = irfft2(self.vk)
        self.hr = 1 + irfft2(self.hk) 
        self.hk = rfft2(self.hr)

        # Add in wave components
        ur_wave = np.zeros_like(self.xx)
        vr_wave = np.zeros_like(self.xx)
        hr_wave = np.zeros_like(self.xx)

        if (np.shape(A) != np.shape(kw)) or (np.shape(A) != np.shape(lw)):
            raise TypeError("Wave amplitudes and wavenumbers must have the same shape")
        
        self.wave_omega =[]
        rng = np.random.default_rng(seed=42)
        # If only one wave component
        if isinstance(A, (float,int)):
            if ((kw != 0) or (lw != 0)) and (A!=0):
                random_phase = rng.uniform(0,2*np.pi)
                self.wave_omega = self.wave_velocities(0,A=A,kw=kw,lw=lw,phase=random_phase)[2]
                hr_wave += self.wave_height(0,A=A,kw=kw,lw=lw,phase=random_phase)[0]
                ur_wave += self.wave_velocities(0,A=A,kw=kw,lw=lw,phase=random_phase)[0]
                vr_wave += self.wave_velocities(0,A=A,kw=kw,lw=lw,phase=random_phase)[1]

        # If multiple wave components    
        else:
            for i, a in enumerate(A):  
                if ((kw[i] != 0) or (lw[i] != 0)) and (a!=0):

                    random_phase = rng.uniform(0,2*np.pi)
                    self.wave_omega.append(self.wave_velocities(0,A=a,kw=kw[i],lw=lw[i],phase=random_phase)[2])
                    hr_wave += self.wave_height(0,A=a,kw=kw[i],lw=lw[i],phase=random_phase)[0]
                    ur_wave += self.wave_velocities(0,A=a,kw=kw[i],lw=lw[i],phase=random_phase)[0]
                    vr_wave += self.wave_velocities(0,A=a,kw=kw[i],lw=lw[i],phase=random_phase)[1]   

        self.ur += ur_wave
        self.vr += vr_wave
        self.hr += hr_wave
        self.uk = rfft2(self.ur)
        self.vk = rfft2(self.vr)
        self.hk = rfft2(self.hr)
        
        print(f'Geostrophic linearity Fr^2/Ro = {self.Fr**2/self.Ro} ')
        if type(A) ==list:
            print(f'Wave linearity A*Ro = {[a*self.Ro for a in A]}')
        else:
            print(f'Wave linearity A*Ro = {A*self.Ro}')
        print(f'Wave omega = {self.wave_omega}')
            
    def set_LM_IC(self,ReL=None, ReL_hyp = 4e13, kernel_type='tophat',kernel_params=None,solve_xi12=True,solve_xi23=True,solve_xi13=True,solve_xi32=True):
        """ Set initial conditions and settings for Lagrangian mean computation.

        Args:
            ReL (float, optional): Reynolds number for Lagrangian equations. Defaults to None.
            ReL_hyp (float, optional): Hyperviscous Reynolds number for Lagrangian equations. Defaults to 4e13.
            kernel_type (str, optional): Type of weight function. Can take values 'tophat,'lowpass', 'lowbandpass' or 'Butterworth' Defaults to 'tophat'.
            kernel_params (dict, optional): Dictionary containing kernel parameters. Defaults to None.
            solve_xi12 (bool, optional): If strategy 1 and true, solves for xi12. Defaults to True.
            solve_xi23 (bool, optional): If strategy 2 and true, solves for xi23. Defaults to False.
            solve_xi13 (bool, optional): If strategy 1 and true, solves for xi13. Defaults to False.
            solve_xi32 (bool, optional): If strategy 3 and true, solves for xi32. Defaults to True.
        """
        if not self.solve_LM:
            print('Not solving Lagrangian equations, skipping this step')
        else:
            # Check for a valid kernel type
            if (kernel_type == 'tophat') or (kernel_type=='lowpass') or (kernel_type=='Butterworth') or (kernel_type=='lowbandpass'):
                self.kernel_type = kernel_type
                self.kernel_params = kernel_params
            elif kernel_type =='exponential': 
                # Special case, make sure we use strategy 2
                if self.strat != 2:
                    print('Changing strategy to 2 for exponential mean')
                    self.strat = 2
                
                self.kernel_type = kernel_type
                self.alpha = kernel_params['alpha']
                if self.Nint > 1:
                    raise ValueError("The exponential mean is found at every timestep. No need to define multiple intervals, it's wasteful")
            else:
                print(f"kernel type '{kernel_type}' doesn't exist. Using 'tophat'")
                self.kernel_type='tophat'
            
            # Initialise current kernel
            self.kernel_current = np.zeros((self.Nint))
            self.kernel_current_int = np.zeros((self.Nint))

            # =================== initialising the Lagrangian mean field quantities ===================

            #  Calculate the viscosity coefficients for the Lagrangian equations
            if (ReL_hyp is None) and (ReL is None):
                self.CcL = 1
                print('Warning: no viscosity for Lagrangian fields')
            elif (ReL_hyp is None):
                self.CcL = np.exp(-self.dt/ReL*self.k2)
            elif (ReL is None):
                self.CcL = np.exp(-self.dt/ReL_hyp*self.k2**4)
            else: 
                self.CcL = np.exp(-self.dt/ReL_hyp*self.k2**4 -self.dt/ReL*self.k2)

            # Setting scalar initialisation
            self.zk = 1j*self.Kxx*self.vk-1j*self.Kyy*self.uk
            self.zr = irfft2(self.zk)
            
            self.z_EMr = np.zeros((self.Nint,self.Ny,self.Nx))
            self.z_LMk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)

            # Setting interpolation initialisation
            self.X_in = np.tile(np.expand_dims(self.xx,0),(self.Nint,1,1))
            self.Y_in = np.tile(np.expand_dims(self.yy,0),(self.Nint,1,1))

            # Setting map initialisation
            # Decide which optional maps to solve
            if self.kernel_type=='exponential':
                self.solve_xi23 = False
            else:
                self.solve_xi23 = solve_xi23
            self.solve_xi32 = solve_xi32
            self.solve_xi13 = solve_xi13
            self.solve_xi12 = solve_xi12

            if self.strat == 1:
                # Maps for strategy 1 
                if self.solve_xi13:
                    self.xi13xk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                    self.xi13yk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                if self.solve_xi12:
                    self.xi12xk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                    self.xi12yk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)

            elif self.strat == 2:
                # Maps for strategy 2
                self.xi21xk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                self.xi21yk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                self.xi21xr = np.zeros((self.Nint,self.Ny,self.Nx))
                self.xi21yr = np.zeros((self.Nint,self.Ny,self.Nx))

                # Initialise the interpolated RHS. Do this outside of RK4 for strategy 2 (no interpolation needed inside RK4 for strategies 1 and 3)
                self.interp_21_uk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                self.interp_21_vk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                self.interp_21_ur = np.zeros((self.Nint,self.Ny,self.Nx))
                self.interp_21_vr = np.zeros((self.Nint,self.Ny,self.Nx))
                self.interp21_scalark = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)

                if self.solve_xi23:
                    self.xi23xk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                    self.xi23yk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                
            elif self.strat == 3:
                # Maps for strategy 3
                self.xi31xk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                self.xi31yk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                if self.solve_xi32:
                    self.xi32xk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                    self.xi32yk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                    self.interp_31_uk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)
                    self.interp_31_vk = np.zeros((self.Nint,self.Ny,int(self.Nx/2+1))).astype(complex)

            else:
                raise ValueError('Strategy must be one of 1,2, or 3.')

    def Butterworth(self,omega,omega_c,n):
        """ Defines a Butterworth filter frequency filter

        Args:
            omega (np.ndarray): array of frequencies over which filter is defined
            omega_c (float): critical frequency of Butterworth filter
            n (int): order of Butterworth filter

        Returns:
            np.ndarray: Butterworth frequency filter 
        """
        return 1/np.sqrt(1+(omega/omega_c)**(2*n))
    
    def _kernel(self,t,kernel_type):
        """ Defines weight function for Lagrangian mean.

        Args:
            t (float): Simulation time
            kernel_type (str): Kernel type. Can be 'tophat','lowpass', 'lowbandpass' or 'Butterworth'.

        Returns:
            list(float,float): [normalised weight function at time t, normalised time integrated weight function at time t]
        """
            
        if kernel_type=='tophat':
            return [1/self.T,t/self.T]
        
        elif kernel_type=='lowpass':
            omega_crit = self.kernel_params['omega_crit']
            t_mid = self.Ntinthf*self.dt
            normalisation = ((1/np.pi)*(sici(omega_crit*(self.T- t_mid))[0] + sici(omega_crit*t_mid)[0]))
            if t==t_mid:
                return [omega_crit/np.pi/normalisation,(1/np.pi)*(sici(omega_crit*(t- t_mid))[0] + sici(omega_crit*t_mid)[0])/normalisation]
            else:
                return [np.sin(omega_crit*(t_mid - t))/np.pi/(t_mid - t)/normalisation,
                        (1/np.pi)*(sici(omega_crit*(t- t_mid))[0] + sici(omega_crit*t_mid)[0])/normalisation]
            
        elif kernel_type=='lowbandpass':
            omega_crit = self.kernel_params['omega_crit']
            omega_crit_band_lower = self.kernel_params['omega_crit_band_lower']
            omega_crit_band_higher = self.kernel_params['omega_crit_band_higher']
            t_mid = self.Ntinthf*self.dt
            normalisation = ((1/np.pi)*((sici(omega_crit*(self.T- t_mid))[0] + sici(omega_crit*t_mid)[0]) + 
                                        (sici(omega_crit_band_higher*(self.T- t_mid))[0] + sici(omega_crit_band_higher*t_mid)[0]) - 
                                        (sici(omega_crit_band_lower*(self.T- t_mid))[0] + sici(omega_crit_band_lower*t_mid)[0])))
            if t==t_mid:
                return [(omega_crit + omega_crit_band_higher - omega_crit_band_lower)/np.pi/normalisation,
                        (1/np.pi)*((sici(omega_crit*(t- t_mid))[0] + sici(omega_crit*t_mid)[0]) + 
                                   (sici(omega_crit_band_higher*(t- t_mid))[0] + sici(omega_crit_band_higher*t_mid)[0]) - 
                                   (sici(omega_crit_band_lower*(t- t_mid))[0] + sici(omega_crit_band_lower*t_mid)[0]))/normalisation]
            else:
                return [(np.sin(omega_crit*(t_mid - t)) + np.sin(omega_crit_band_higher*(t_mid - t)) - np.sin(omega_crit_band_lower*(t_mid - t)))/np.pi/(t_mid - t)/normalisation,
                        (1/np.pi)*((sici(omega_crit*(t- t_mid))[0] + sici(omega_crit*t_mid)[0]) +
                                   (sici(omega_crit_band_higher*(t- t_mid))[0] + sici(omega_crit_band_higher*t_mid)[0]) - 
                                   (sici(omega_crit_band_lower*(t- t_mid))[0] + sici(omega_crit_band_lower*t_mid)[0]))/normalisation]
               
        elif kernel_type == 'Butterworth':
            omega_crit = self.kernel_params['omega_crit']
            n = self.kernel_params['n']
            t_mid = self.Ntinthf*self.dt

            omega = np.linspace(-omega_crit*10,omega_crit*10,1000)
            kernel_nonorm = np.real(np.trapz((1/2/np.pi)*self.Butterworth(omega,omega_crit,n)*np.exp(1j*omega*(t_mid - t)),omega))
            kernel_int_nonorm = np.real(np.trapz((1j/2/np.pi/omega)*self.Butterworth(omega,omega_crit,n)
                                         *np.exp(1j*omega*t_mid)*(np.exp(-1j*omega*t) - 1),omega))
            normalisation = np.real(np.trapz((1j/2/np.pi/omega)*self.Butterworth(omega,omega_crit,n)
                                         *np.exp(1j*omega*t_mid)*(np.exp(-1j*omega*self.T) - 1),omega))
            return [kernel_nonorm/normalisation,
                        kernel_int_nonorm/normalisation]

    def wave_velocities(self,t,A=0.5,kw=1,lw=0,phase=0):
        """Creates analytic wave velocity with amplitude A, wavenumbers kw, lw, and given phase. 

        Args:
            t (float): Time at which to give the wave field
            A (float, optional): Wave amplitude. Defaults to 0.5.
            kw (int, optional): Wave x-wavenumber. Defaults to 1.
            lw (int, optional): Wave y-wavenumber. Defaults to 0.
            phase (int, optional): Wave phase. Defaults to 0.

        Returns:
            np.ndarray: real wave x-velocity component
            np.ndarray: real wave y-velocity component
            float: wave frequency
        """

        wave_omega = np.sqrt(self.Ro**-2+self.Fr**-2*(kw**2 + lw**2))
        ur_wave = wave_omega*kw*self.Ro/(kw**2 + lw**2)*A*np.cos(kw*self.xx + lw*self.yy + phase - wave_omega*t) - lw*A/(kw**2 + lw**2)*np.sin(kw*self.xx + lw*self.yy + phase - wave_omega*t)
        vr_wave = wave_omega*lw*self.Ro/(kw**2 + lw**2)*A*np.cos(kw*self.xx + lw*self.yy + phase - wave_omega*t) + kw*A/(kw**2 + lw**2)*np.sin(kw*self.xx + lw*self.yy + phase - wave_omega*t)

        return ur_wave, vr_wave, wave_omega
    
    def wave_displacement(self,t,A=0.5,kw=1,lw=0,phase=0):
        """Creates analytic wave displacement with amplitude A, wavenumbers kw, lw, and given phase. 

        Args:
            t (float): Time at which to give the wave field
            A (float, optional): Wave amplitude. Defaults to 0.5.
            kw (int, optional): Wave x-wavenumber. Defaults to 1.
            lw (int, optional): Wave y-wavenumber. Defaults to 0.
            phase (int, optional): Wave phase. Defaults to 0.

        Returns:
            np.ndarray: real wave x-displacement component
            np.ndarray: real wave y-displacement component
            float: wave frequency
        """

        wave_omega = np.sqrt(self.Ro**-2+self.Fr**-2*(kw**2 + lw**2))
        dxr_wave = -kw*self.Ro/(kw**2 + lw**2)*A*np.sin(kw*self.xx + lw*self.yy + phase - wave_omega*t) - lw*A/wave_omega/(kw**2 + lw**2)*np.cos(kw*self.xx + lw*self.yy + phase - wave_omega*t)
        dyr_wave = -lw*self.Ro/(kw**2 + lw**2)*A*np.sin(kw*self.xx + lw*self.yy + phase - wave_omega*t) + kw*A/wave_omega/(kw**2 + lw**2)*np.cos(kw*self.xx + lw*self.yy + phase - wave_omega*t)

        return dxr_wave, dyr_wave, wave_omega
    def wave_height(self,t,A=0.5,kw=1,lw=0,phase=0):
        """Creates an analytic wave (height) with amplitude A, wavenumbers kw, lw, and given phase. 

        Args:
            t (float): Time at which to give the wave field
            A (float, optional): Wave amplitude. Defaults to 0.5.
            kw (int, optional): Wave x-wavenumber. Defaults to 1.
            lw (int, optional): Wave y-wavenumber. Defaults to 0.
            phase (int, optional): Wave phase. Defaults to 0.

        Returns:
            np.ndarray: real wave height
            float: wave frequency
        """

        wave_omega = np.sqrt(self.Ro**-2+self.Fr**-2*(kw**2 + lw**2))
        hr_wave = self.Ro*A*np.cos(kw*self.xx + lw*self.yy + phase - wave_omega*t) 
        return hr_wave, wave_omega
    
    
    
    def _RHS_SW(self,uk,vk,hk):
        """ Computes RHS of shallow water equations in spectral space given u, v, h in spectral space.

        Args:
            uk (np.ndarray): Spectral x-velocity
            vk (np.ndarray): Spectral y-velocity
            hk (np.ndarray): Spectral height

        Returns:
            tuple(np.ndarray,np.ndarray,np.ndarray): spectral RHS of u,v,h equations
        """
        ur = irfft2(uk)  
        vr = irfft2(vk)
        hr = irfft2(hk)
        
        # Option to remove all nonlinear terms (for testing)
        if self.linear:
            RHSuk = (vk/self.Ro - 1j * self.Kxx * hk/self.Fr**2)
            RHSvk = (-uk/self.Ro - 1j * self.Kyy * hk/self.Fr**2)
            RHShk = (-1j * self.Kxx * uk - 1j * self.Kyy * vk)
                

        else:
            if self.MSW:
                
                # Implements Buhler 1998 modified shallow water by changing the form of the pressure 
                # term so that shocks aren't generated by nonlinear IGWs
                u_xr = irfft2(1j * self.Kxx * uk)
                v_xr = irfft2(1j * self.Kxx * vk)
                u_yr = irfft2(1j * self.Kyy * uk)
                v_yr = irfft2(1j * self.Kyy * vk)
                h_xr = irfft2(1j * self.Kxx * hk)
                h_yr = irfft2(1j * self.Kyy * hk)
                
                # Implement the pressure term with the other nonlinear terms
                Nu = self.L * rfft2(ur * u_xr + vr * u_yr + (1/self.Fr**2/hr**3)*h_xr)
                Nv = self.L * rfft2(ur * v_xr + vr * v_yr + (1/self.Fr**2/hr**3)*h_yr)
                Nhx = self.L * rfft2(ur * hr)
                Nhy = self.L * rfft2(vr * hr)

                RHSuk = (-Nu + vk/self.Ro )
                RHSvk = (-Nv - uk/self.Ro )
                RHShk = (-1j * self.Kxx * Nhx - 1j * self.Kyy * Nhy)

            else:
                # Standard shallow water
                u_xr = irfft2(1j * self.Kxx * uk)
                v_xr = irfft2(1j * self.Kxx * vk)
                u_yr = irfft2(1j * self.Kyy * uk)
                v_yr = irfft2(1j * self.Kyy * vk)

                Nu = self.L * rfft2(ur * u_xr + vr * u_yr)
                Nv = self.L * rfft2(ur * v_xr + vr * v_yr)
                Nhx = self.L * rfft2(ur * hr)
                Nhy = self.L * rfft2(vr * hr)

                RHSuk = (-Nu + vk/self.Ro - 1j * self.Kxx * hk/self.Fr**2)
                RHSvk = (-Nv - uk/self.Ro - 1j * self.Kyy * hk/self.Fr**2)
                RHShk = (-1j * self.Kxx * Nhx - 1j * self.Kyy * Nhy)
                
        return RHSuk, RHSvk, RHShk
        
    
    def _timestep_RK4_SW_eqn(self):
        """ Takes one RK4 timestep of the momentum equations, updates spectral and real u,v,h
        """
        
        # First find k1 
        k1uk, k1vk, k1hk = self._RHS_SW(self.uk,self.vk,self.hk)
        
        # Then evaluate RHS at time t + dt/2 using k1 estimate for gradient
        k2uk, k2vk, k2hk = self._RHS_SW(self.uk + k1uk*self.dt/2,self.vk + k1vk*self.dt/2,self.hk+ k1hk*self.dt/2)

        # Then evaluate RHS at time t + dt/2 using k2 estimate for gradient
        k3uk, k3vk, k3hk = self._RHS_SW(self.uk + k2uk*self.dt/2,self.vk + k2vk*self.dt/2,self.hk+ k2hk*self.dt/2)

        # Then evaluate RHS at time t + dt using k3 estimate for gradient
        k4uk, k4vk, k4hk = self._RHS_SW(self.uk + k3uk*self.dt,self.vk + k3vk*self.dt,self.hk+ k3hk*self.dt)

        # Weighted average slope approximation
        muk = (k1uk + 2*k2uk + 2*k3uk + k4uk)/6
        mvk = (k1vk + 2*k2vk + 2*k3vk + k4vk)/6
        mhk = (k1hk + 2*k2hk + 2*k3hk + k4hk)/6
        
        # Perform timestep
        self.uk = self.Cc * (self.uk + muk*self.dt)
        self.vk = self.Cc * (self.vk + mvk*self.dt)
        self.hk = mhk*self.dt + self.hk
        
        # Update real fields
        self.ur = irfft2(self.uk)
        self.vr = irfft2(self.vk)
        self.hr = irfft2(self.hk)

    
    def _RHS_xi31(self,interval):
        """ Calculates RHS of xi31 map equation. Doesn't need field inputs as not used with RK4.

        Args:
            interval (int): interval number to solve for

        Returns:
            tuple(np.ndarray,np.ndarray): x and y components of xi31 RHS for the given interval in spectral space
        """
        # First half of interval
        if self.flag[interval] == 1 or self.flag[interval] == 0:
            RHSxi31xk = 0
            RHSxi31yk = 0
            
        # Second half of interval
        elif self.flag[interval] ==2:
            # prepare fields for the next time step
            xi31xr = irfft2(self.xi31xk[interval,:,:])
            xi31yr = irfft2(self.xi31yk[interval,:,:])
            CurPosX = self.xx + xi31xr
            CurPosY = self.yy + xi31yr

            # These are the positions at which we find the scalar, so assign as attributes
            self.X_in[interval,:,:] = CurPosX - np.floor(CurPosX/(2*np.pi))*(2*np.pi)
            self.Y_in[interval,:,:] = CurPosY - np.floor(CurPosY/(2*np.pi))*(2*np.pi)
            
            RHSxi31xk = self._interp(self.ur,self.X_in[interval,:,:],self.Y_in[interval,:,:])[1]
            RHSxi31yk = self._interp(self.vr,self.X_in[interval,:,:],self.Y_in[interval,:,:])[1]

            if self.solve_xi32:
                self.interp_31_uk[interval,:,:] = RHSxi31xk
                self.interp_31_vk[interval,:,:] = RHSxi31yk

        return RHSxi31xk, RHSxi31yk
        
    
    def _RHS_xi21(self,interval,xi21xk,xi21yk):
        """ Calculates RHS of xi21 map equation. Note that the interpolation happens outside of this function to avoid interpolating on every RK4 step.

        Args:
            interval (int): interval number to solve for
            xi21xk (np.ndarray): x-component of xi21 map in spectral space for given interval
            xi21yk (np.ndarray): y-component of xi21 map in spectral space for given interval

        Returns:
            tuple(np.ndarray,np.ndarray): x and y components of xi21 RHS in spectral space for given interval
        """
        xi21xr = irfft2(xi21xk)
        xi21yr = irfft2(xi21yk)

        # Generating nonlinear terms for x map
        xi21x_xr = irfft2(1j * self.Kxx * xi21xk)
        xi21x_yr = irfft2(1j * self.Kyy * xi21xk)

        # Generating nonlinear terms for y map
        xi21y_xr = irfft2(1j * self.Kxx * xi21yk)
        xi21y_yr = irfft2(1j * self.Kyy * xi21yk)

        if self.kernel_type == 'exponential':
            Nxixk = self.L * rfft2(xi21xr * xi21x_xr + xi21yr * xi21x_yr)
            Nxiyk = self.L * rfft2(xi21xr * xi21y_xr + xi21yr * xi21y_yr)
            
            RHSxi21xk = (-(Nxixk + xi21xk)*self.alpha + self.interp_21_uk[interval,:,:])
            RHSxi21yk = (-(Nxiyk + xi21yk)*self.alpha + self.interp_21_vk[interval,:,:])
        else:
            Nxixk = self.L * (1 - self.kernel_current_int[interval]) * rfft2(self.interp_21_ur[interval,:,:] * xi21x_xr + self.interp_21_vr[interval,:,:] * xi21x_yr)
            Nxiyk = self.L * (1 - self.kernel_current_int[interval]) * rfft2(self.interp_21_ur[interval,:,:] * xi21y_xr + self.interp_21_vr[interval,:,:] * xi21y_yr)

            RHSxi21xk = (self.interp_21_uk[interval,:,:]*self.kernel_current_int[interval] - Nxixk)
            RHSxi21yk = (self.interp_21_vk[interval,:,:]*self.kernel_current_int[interval] - Nxiyk)

        return RHSxi21xk, RHSxi21yk

    def _RHS_xi23(self,interval,xi23xk,xi23yk):
        """ Calculates RHS of xi23 map equation. Note that the interpolation happens outside of this function to avoid interpolating on every RK4 step.

        Args:
            interval (int): interval number to solve for
            xi23xk (np.ndarray): x-component of xi23 map in spectral space for given interval
            xi23yk (np.ndarray): y-component of xi23 map in spectral space for given interval

        Returns:
            tuple(np.ndarray,np.ndarray): x and y components of xi23 RHS in spectral space for given interval
        """

        # Generating nonlinear terms for x map
        xi23x_xr = irfft2(1j * self.Kxx * xi23xk)
        xi23x_yr = irfft2(1j * self.Kyy * xi23xk)
        Nxixk = self.L * (1 - self.kernel_current_int[interval]) * rfft2(self.interp_21_ur[interval,:,:] * xi23x_xr + self.interp_21_vr[interval,:,:] * xi23x_yr)

        # Generating nonlinear terms for y map
        xi23y_xr = irfft2(1j * self.Kxx * xi23yk)
        xi23y_yr = irfft2(1j * self.Kyy * xi23yk)
        Nxiyk = self.L * (1 - self.kernel_current_int[interval]) * rfft2(self.interp_21_ur[interval,:,:] * xi23y_xr + self.interp_21_vr[interval,:,:] * xi23y_yr)


        if self.flag[interval] == 1:

            RHSxi23xk = self.interp_21_uk[interval,:,:]*self.kernel_current_int[interval] - Nxixk
            RHSxi23yk = self.interp_21_vk[interval,:,:]*self.kernel_current_int[interval] - Nxiyk

        elif self.flag[interval] == 2:

            RHSxi23xk = self.interp_21_uk[interval,:,:]*(self.kernel_current_int[interval] - 1) - Nxixk
            RHSxi23yk = self.interp_21_vk[interval,:,:]*(self.kernel_current_int[interval] - 1) - Nxiyk

        else:
            RHSxi23xk = np.zeros_like(xi23xk)
            RHSxi23yk = np.zeros_like(xi23yk)

        return RHSxi23xk, RHSxi23yk

    def _RHS_xi13(self,interval,xi13xk,xi13yk):
        """ Calculates RHS of xi13 map equation

        Args:
            interval (int): interval number to solve for
            xi13xk (np.ndarray): x-component of xi13 map in spectral space for given interval
            xi13yk (np.ndarray): y-component of xi13 map in spectral space for given interval

        Returns:
            tuple(np.ndarray,np.ndarray): x and y components of xi13 RHS in spectral space for given interval
        """

        # Solving the equation for x map
        xi13x_xr = irfft2(1j * self.Kxx * xi13xk)
        xi13x_yr = irfft2(1j * self.Kyy * xi13xk)
        NzLx = self.L * rfft2(self.ur * xi13x_xr + self.vr * xi13x_yr)

        # Solving the equation for y map
        xi13y_xr = irfft2(1j * self.Kxx * xi13yk)
        xi13y_yr = irfft2(1j * self.Kyy * xi13yk)
        NzLy = self.L * rfft2(self.ur * xi13y_xr + self.vr * xi13y_yr)

        if self.flag[interval] == 1:
            RHSxi13xk = -NzLx
            RHSxi13yk = -NzLy

        elif self.flag[interval] == 2:    
        
            RHSxi13xk = (-NzLx  - self.uk)
            RHSxi13yk = (-NzLy  - self.vk)

        else:
            RHSxi13xk = np.zeros_like(NzLx)
            RHSxi13yk = np.zeros_like(NzLy)

        return RHSxi13xk, RHSxi13yk
    
    
    def _RHS_xi12(self,interval,xi12xk,xi12yk):
        """ Calculates RHS of xi12 map equation

        Args:
            interval (int): interval number to solve for
            xi12xk (np.ndarray): x-component of xi12 map in spectral space for given interval
            xi12yk (np.ndarray): y-component of xi12 map in spectral space for given interval

        Returns:
            tuple(np.ndarray,np.ndarray): x and y components of xi12 RHS in spectral space for given interval
        """
        
        # Solving the equation for x map
        xi12x_xr = irfft2(1j * self.Kxx * xi12xk)
        xi12x_yr = irfft2(1j * self.Kyy * xi12xk)
        NzL = self.L * rfft2(self.ur * xi12x_xr + self.vr * xi12x_yr)

        RHSxi12xk = -NzL  - self.uk*self.kernel_current_int[interval]

        # Solving the equation for y map
        xi12y_xr = irfft2(1j * self.Kxx * xi12yk)
        xi12y_yr = irfft2(1j * self.Kyy * xi12yk)
        NzL = self.L * rfft2(self.ur * xi12y_xr + self.vr * xi12y_yr)

        RHSxi12yk = -NzL  - self.vk*self.kernel_current_int[interval]

        return RHSxi12xk, RHSxi12yk
    
    def _RHS_xi32(self,interval,xi32xk,xi32yk):
        """ Calculates RHS of xi32 map equation

        Args:
            interval (int): interval number to solve for
            xi32xk (np.ndarray): x-component of xi32 map in spectral space for given interval
            xi32yk (np.ndarray): y-component of xi32 map in spectral space for given interval

        Returns:
            tuple(np.ndarray,np.ndarray): x and y components of xi32 RHS in spectral space for given interval
        """
        
        if self.flag[interval] == 1:

            # Solving the equation for x map
            xi32x_xr = irfft2(1j * self.Kxx * xi32xk)
            xi32x_yr = irfft2(1j * self.Kyy * xi32xk)
            NzL = self.L * rfft2(self.ur * xi32x_xr + self.vr * xi32x_yr)

            RHSxi32xk = -NzL  - self.uk*self.kernel_current_int[interval]

            # Solving the equation for y map
            xi32y_xr = irfft2(1j * self.Kxx * xi32yk)
            xi32y_yr = irfft2(1j * self.Kyy * xi32yk)
            NzL = self.L * rfft2(self.ur * xi32y_xr + self.vr * xi32y_yr)

            RHSxi32yk = -NzL  - self.vk*self.kernel_current_int[interval]

        elif self.flag[interval] ==2:
            RHSxi32xk = (1 - self.kernel_current_int[interval])*self.interp_31_uk[interval,:,:]
            RHSxi32yk = (1 - self.kernel_current_int[interval])*self.interp_31_vk[interval,:,:]

        else:
            RHSxi32xk = np.zeros_like(xi32xk)
            RHSxi32yk = np.zeros_like(xi32yk)

        return RHSxi32xk, RHSxi32yk

    def _RHS_scalar(self,interval,scalarLM_k,scalar_k):
        """ Calculates RHS of partial scalar mean equation

        Args:
            interval (int): interval number to solve for
            scalarLM_k (np.ndarray): Partial Lagrangian mean of scalar in spectral space for given interval
            scalar_k (np.ndarray): Scalar in spectral space

        Returns:
            np.ndarray: RHS of partial scalar mean equation for given interval
        """
        
        if self.strat ==1 or ((self.strat ==3) and (self.flag[interval] == 1)):
            g_LM_xr = irfft2(1j * self.Kxx * scalarLM_k)
            g_LM_yr = irfft2(1j * self.Kyy * scalarLM_k)
            NzL = self.L * rfft2(self.ur * g_LM_xr + self.vr * g_LM_yr)

            RHSgk = (-NzL + scalar_k*self.kernel_current[interval])
            
        elif (self.flag[interval] == 2) and (self.strat==3):
            scalar_r = irfft2(scalar_k)
            RHSgk = self._interp(scalar_r,self.X_in[interval,:,:],self.Y_in[interval,:,:])[1]*self.kernel_current[interval]

        elif self.strat ==2:
            g_LM_xr = irfft2(1j * self.Kxx * scalarLM_k)
            g_LM_yr = irfft2(1j * self.Kyy * scalarLM_k)
            
            if self.kernel_type == 'exponential':
                NzL = self.L * rfft2(self.xi21xr[interval,:,:] * g_LM_xr + self.xi21yr[interval,:,:] * g_LM_yr)
                RHSgk = -NzL*self.alpha + (self.interp21_scalark[interval,:,:] -scalarLM_k)*self.alpha
            else:
                NzL = self.L * (1 - self.kernel_current_int[interval])*rfft2(self.interp_21_ur[interval,:,:] * g_LM_xr + self.interp_21_vr[interval,:,:] * g_LM_yr)
                RHSgk = - NzL + self.interp21_scalark[interval,:,:]*self.kernel_current[interval]

        return RHSgk
    
    def _timestep_xi31(self,interval): 
        """ Takes a timestep of xi31 equation (no RK4 as no advection term) for given interval
        
        Args:
            interval (int): interval number to solve for
        """
        
        # No advection term, so we don't use RK4 here
        k1kx, k1ky = self._RHS_xi31(interval)
        
        # Perform timestep
        self.xi31xk[interval,:,:] = self.CcL * (self.xi31xk[interval,:,:] + k1kx*self.dt)
        self.xi31yk[interval,:,:] = self.CcL * (self.xi31yk[interval,:,:] + k1ky*self.dt)

    
    def _timestep_RK4_xi32(self,interval): 
        """ Takes an RK4 timestep of the xi32 equation in first half of interval, and a simple timestep in second half of interval (no advection)
        
        Args:
            interval (int): interval number to solve for
        """
        # Only do RK4 when there's an advection term:
        if self.flag[interval] ==1:
            k1gkx, k1gky = self._RHS_xi32(interval,self.xi32xk[interval,:,:],self.xi32yk[interval,:,:])
            
            k2gkx, k2gky = self._RHS_xi32(interval,self.xi32xk[interval,:,:] + k1gkx*self.dt/2, self.xi32yk[interval,:,:] + k1gky*self.dt/2)

            k3gkx, k3gky = self._RHS_xi32(interval,self.xi32xk[interval,:,:] + k2gkx*self.dt/2, self.xi32yk[interval,:,:] + k2gky*self.dt/2)

            k4gkx, k4gky = self._RHS_xi32(interval,self.xi32xk[interval,:,:] + k3gkx*self.dt, self.xi32yk[interval,:,:] + k3gky*self.dt)

            # Weighted average slope approximation
            mgkx = (k1gkx + 2*k2gkx + 2*k3gkx + k4gkx)/6
            mgky = (k1gky + 2*k2gky + 2*k3gky + k4gky)/6
            
        elif self.flag[interval] ==2:
            mgkx, mgky = self._RHS_xi32(interval,self.xi32xk[interval,:,:],self.xi32yk[interval,:,:])

        # Perform timestep
        self.xi32xk[interval,:,:] = self.CcL * (self.xi32xk[interval,:,:] + mgkx*self.dt)
        self.xi32yk[interval,:,:] = self.CcL * (self.xi32yk[interval,:,:] + mgky*self.dt)

    def _timestep_RK4_xi13(self,interval): 
        """ Takes an RK4 timestep of the xi31 equation for given interval
        
        Args:
            interval (int): interval number to solve for
        """
       
        k1gkx, k1gky = self._RHS_xi13(interval,self.xi13xk[interval,:,:],self.xi13yk[interval,:,:])
        
        k2gkx, k2gky = self._RHS_xi13(interval,self.xi13xk[interval,:,:] + k1gkx*self.dt/2, self.xi13yk[interval,:,:] + k1gky*self.dt/2)

        k3gkx, k3gky = self._RHS_xi13(interval,self.xi13xk[interval,:,:] + k2gkx*self.dt/2, self.xi13yk[interval,:,:] + k2gky*self.dt/2)

        k4gkx, k4gky = self._RHS_xi13(interval,self.xi13xk[interval,:,:] + k3gkx*self.dt, self.xi13yk[interval,:,:] + k3gky*self.dt)

        # Weighted average slope approximation
        mgkx = (k1gkx + 2*k2gkx + 2*k3gkx + k4gkx)/6
        mgky = (k1gky + 2*k2gky + 2*k3gky + k4gky)/6
            
        # Perform timestep
        self.xi13xk[interval,:,:] = self.CcL * (self.xi13xk[interval,:,:] + mgkx*self.dt)
        self.xi13yk[interval,:,:] = self.CcL * (self.xi13yk[interval,:,:] + mgky*self.dt)

    def _timestep_RK4_xi12(self,interval): 
        """ Takes an RK4 timestep of the xi12 equation for given interval
        
        Args:
            interval (int): interval number to solve for
        """
        
        k1gkx, k1gky = self._RHS_xi12(interval,self.xi12xk[interval,:,:],self.xi12yk[interval,:,:])
        
        k2gkx, k2gky = self._RHS_xi12(interval,self.xi12xk[interval,:,:] + k1gkx*self.dt/2, self.xi12yk[interval,:,:] + k1gky*self.dt/2)

        k3gkx, k3gky = self._RHS_xi12(interval,self.xi12xk[interval,:,:] + k2gkx*self.dt/2, self.xi12yk[interval,:,:] + k2gky*self.dt/2)

        k4gkx, k4gky = self._RHS_xi12(interval,self.xi12xk[interval,:,:] + k3gkx*self.dt, self.xi12yk[interval,:,:] + k3gky*self.dt)

        # Weighted average slope approximation
        mgkx = (k1gkx + 2*k2gkx + 2*k3gkx + k4gkx)/6
        mgky = (k1gky + 2*k2gky + 2*k3gky + k4gky)/6
            
        # Perform timestep
        self.xi12xk[interval,:,:] = self.CcL * (self.xi12xk[interval,:,:] + mgkx*self.dt)
        self.xi12yk[interval,:,:] = self.CcL * (self.xi12yk[interval,:,:] + mgky*self.dt)

    def _timestep_RK4_xi21(self,interval): 
        """ Takes an RK4 timestep of the xi21 equation for given interval
        
        Args:
            interval (int): interval number to solve for
        """
        
        # Update RHS interpolated term, do this outside of RK4 loop:
        self.interp_21_ur[interval,:,:], self.interp_21_uk[interval,:,:] = self._interp(self.ur,self.X_in[interval,:,:],self.Y_in[interval,:,:])
        self.interp_21_vr[interval,:,:], self.interp_21_vk[interval,:,:] = self._interp(self.vr,self.X_in[interval,:,:],self.Y_in[interval,:,:])

        k1gkx, k1gky = self._RHS_xi21(interval,self.xi21xk[interval,:,:],self.xi21yk[interval,:,:])
        
        k2gkx, k2gky = self._RHS_xi21(interval,self.xi21xk[interval,:,:] + k1gkx*self.dt/2, self.xi21yk[interval,:,:] + k1gky*self.dt/2)

        k3gkx, k3gky = self._RHS_xi21(interval,self.xi21xk[interval,:,:] + k2gkx*self.dt/2, self.xi21yk[interval,:,:] + k2gky*self.dt/2)

        k4gkx, k4gky = self._RHS_xi21(interval,self.xi21xk[interval,:,:] + k3gkx*self.dt, self.xi21yk[interval,:,:] + k3gky*self.dt)

        # Weighted average slope approximation
        mgkx = (k1gkx + 2*k2gkx + 2*k3gkx + k4gkx)/6
        mgky = (k1gky + 2*k2gky + 2*k3gky + k4gky)/6
            
        # Perform timestep
        self.xi21xk[interval,:,:] = self.CcL * (self.xi21xk[interval,:,:] + mgkx*self.dt)
        self.xi21yk[interval,:,:] = self.CcL * (self.xi21yk[interval,:,:] + mgky*self.dt)

        # Prepare fields for the next time step. We do this outside of the _RHS_xi21 function for strategy 2 to avoid RK4 stepping the interpolation.
        self.xi21xr[interval,:,:] = irfft2(self.xi21xk[interval,:,:])
        self.xi21yr[interval,:,:] = irfft2(self.xi21yk[interval,:,:])
        CurPosX = self.xx + self.xi21xr[interval,:,:]
        CurPosY = self.yy + self.xi21yr[interval,:,:]

        # These are the positions at which we find the scalar, so assign as attributes
        self.X_in[interval,:,:] = CurPosX - np.floor(CurPosX/(2*np.pi))*(2*np.pi)
        self.Y_in[interval,:,:] = CurPosY - np.floor(CurPosY/(2*np.pi))*(2*np.pi)

    def _timestep_RK4_xi23(self,interval): 
        """ Takes an RK4 timestep of the xi23 equation for given interval
        
        Args:
            interval (int): interval number to solve for
        """
        
        k1gkx, k1gky = self._RHS_xi23(interval,self.xi23xk[interval,:,:],self.xi23yk[interval,:,:])
        
        k2gkx, k2gky = self._RHS_xi23(interval,self.xi23xk[interval,:,:] + k1gkx*self.dt/2, self.xi23yk[interval,:,:] + k1gky*self.dt/2)

        k3gkx, k3gky = self._RHS_xi23(interval,self.xi23xk[interval,:,:] + k2gkx*self.dt/2, self.xi23yk[interval,:,:] + k2gky*self.dt/2)

        k4gkx, k4gky = self._RHS_xi23(interval,self.xi23xk[interval,:,:] + k3gkx*self.dt, self.xi23yk[interval,:,:] + k3gky*self.dt)

        # Weighted average slope approximation
        mgkx = (k1gkx + 2*k2gkx + 2*k3gkx + k4gkx)/6
        mgky = (k1gky + 2*k2gky + 2*k3gky + k4gky)/6
            
        # Perform timestep
        self.xi23xk[interval,:,:] = self.CcL * (self.xi23xk[interval,:,:] + mgkx*self.dt)
        self.xi23yk[interval,:,:] = self.CcL * (self.xi23yk[interval,:,:] + mgky*self.dt)
        
    def _timestep_RK4_scalar(self,interval,scalar_k,scalarLM_k):
        """ Takes an RK4 timestep of the partial scalar mean equation for given interval

        Args:
            interval (int): interval number to solve for
            scalar_k (np.ndarray): Scalar in spectral space
            scalarLM_k (np.ndarray): Partial Lagrangian mean of scalar in spectral space for given interval
            
        Returns:
            np.ndarray: Updated scalar in spectral space for given interval
        """
        if self.strat ==2:
            scalar_r = irfft2(scalar_k)
            self.interp21_scalark[interval,:,:] = self._interp(scalar_r,self.X_in[interval,:,:],self.Y_in[interval,:,:])[1]
    
        if (self.strat ==1) or (self.strat==2) or ((self.strat ==3) and (self.flag[interval] == 1)):
            
            # Only use RK4 on the advection term
            k1gk = self._RHS_scalar(interval,scalarLM_k,scalar_k)
            
            k2gk = self._RHS_scalar(interval,scalarLM_k + k1gk*self.dt/2,scalar_k)

            k3gk = self._RHS_scalar(interval,scalarLM_k + k2gk*self.dt/2,scalar_k)

            k4gk = self._RHS_scalar(interval,scalarLM_k + k3gk*self.dt,scalar_k)

            # Weighted average slope approximation
            mgk = (k1gk + 2*k2gk + 2*k3gk + k4gk)/6

        elif (self.flag[interval] == 2) and (self.strat==3):
            mgk = self._RHS_scalar(interval,scalarLM_k,scalar_k)
        
        # Perform timestep
        scalarLM_k = self.CcL * (scalarLM_k + mgk*self.dt)
        return scalarLM_k


    def _do_timesteps(self, start_step, end_step):
        """ Exectutes timesteps of shallow water and Lagrangian mean equations

        Args:
            start_step (int): start timestep of timesteps
            end_step (int): end timestep of timesteps
        """
        # Initialise working intervals:
        t = (start_step + 0.5)*self.dt
        intervals = []
        next_int_to_finish = 0
        next_int_to_second_half = 0
        for i in range(self.Nint)[::-1]:
            if (t < self.interval_start[i] + self.T/2) and (t > self.interval_start[i]):
                intervals.append(i)
                self.flag[i] = 1
                next_int_to_second_half = i
            elif (t > self.interval_start[i] + self.T/2) and (t < self.interval_start[i] + self.T):
                intervals.append(i)
                self.flag[i] = 2
                next_int_to_finish = i
        if len(intervals) > 0:    
            next_int_to_start = max(intervals) + 1
            print(f'Intervals being solved: {min(intervals)}-{max(intervals)}, time = {self.dt*self.timestep:.2f}')
        else:
            print(f'No LM currently being solved')
            
        
        # Loop over timesteps
        for iTime in tqdm(range(start_step, end_step)):
            t = (iTime + 0.5)*self.dt # midpoint of timestep
            self.t = t
            self.timestep +=1
            if len(intervals) > 0:
                # Update working intervals
                if (next_int_to_finish < self.Nint-1) and (t > self.interval_start[next_int_to_finish] + self.T):
                    self.flag[next_int_to_finish] = 0
                    next_int_to_finish += 1
                    intervals.pop()
                    print(f'Intervals updated to: {min(intervals)}-{max(intervals)}, time = {self.dt*self.timestep:.2f}')
                if (next_int_to_second_half < self.Nint-1) and (t > self.interval_start[next_int_to_second_half] + self.T/2):
                    self.flag[next_int_to_second_half] = 2
                    next_int_to_second_half +=1
                if (next_int_to_start < self.Nint-1) and (t > self.interval_start[next_int_to_start]):
                    self.flag[next_int_to_start] = 1
                    intervals.insert(0,next_int_to_start)
                    next_int_to_start +=1
                    print(f'Intervals updated to: {min(intervals)}-{max(intervals)}, time = {self.dt*self.timestep:.2f}')

            # Timestep momentum equation
            self._timestep_RK4_SW_eqn()

            self.zk = 1j*self.Kxx*self.vk-1j*self.Kyy*self.uk
            self.zr = irfft2(self.zk)

            if self.save_inst and self.timestep % self.Nt_save_inst == 0:
                self.timesteps_inst.append(self.timestep)
                
                self.z_ts[len(self.timesteps_inst)-1,:] = self.zr[self.ny_inst,:] 
                self.u_ts[len(self.timesteps_inst)-1,:] = self.ur[self.ny_inst,:]
                self.v_ts[len(self.timesteps_inst)-1,:] = self.vr[self.ny_inst,:]
                if self.solve_LM and (self.kernel_type == 'exponential'):
                    self.z_LMr = irfft2(self.z_LMk)
                    self.z_LM_ts[len(self.timesteps_inst)-1,:] = self.z_LMr[0,int(self.Ny/2),:]
                    self.z_EM_ts[len(self.timesteps_inst)-1,:] = self.z_EMr[0,int(self.Ny/2),:]
            if self.solve_LM:

                for interval in intervals:
                    if self.kernel_type !='exponential':
                        # Define kernels at time t to avoid calculating them all the time
                        self.kernel_current[interval] = self._kernel(t - self.interval_start[interval], kernel_type = self.kernel_type)[0]
                        self.kernel_current_int[interval] = self._kernel(t - self.interval_start[interval], kernel_type = self.kernel_type)[1]

                    # Timestep the maps
                    if self.strat == 1:
                        if self.solve_xi13:
                            self._timestep_RK4_xi13(interval)
                        if self.solve_xi12:
                            self._timestep_RK4_xi12(interval)
                    elif self.strat == 2:
                        self._timestep_RK4_xi21(interval)
                        if self.solve_xi23:
                            self._timestep_RK4_xi23(interval)
                    elif self.strat == 3:
                        self._timestep_xi31(interval)
                        if self.solve_xi32:
                            self._timestep_RK4_xi32(interval)

                    # Now solve the Lagrangian mean PDEs for scalar z
                    self.z_LMk[interval,:,:] = self._timestep_RK4_scalar(interval,self.zk,self.z_LMk[interval,:,:])                

                    # adding fields to get the Eulerian mean, define with the kernel
                    if self.kernel_type != 'exponential':
                        self.z_EMr[interval,:,:] += self.zr*self.kernel_current[interval]*self.dt
                    else:
                        self.z_EMr[interval,:,:] = self.z_EMr[interval,:,:]*np.exp(-self.alpha*self.dt) + self.alpha*self.dt*self.zr  
            
            
            
            # Make a movie    
            if self.movie_type is not None:
                
                if iTime % self.movie_freq == 0:
                    
                    if self.movie_type =='vorticity':
                        self.plot_vorticity(iTime)
                    if self.movie_type =='pv':
                        self.plot_pv(iTime)
                    if self.movie_type =='u velocity':
                        self.plot_u(iTime)
                    if self.movie_type =='v velocity':
                        self.plot_v(iTime)
                    elif self.movie_type =='vorticity means':
                        self.plot_vorticity_means(iTime)
                    elif self.movie_type =='jacobian':
                        self.plot_jacobian(iTime)

            
    def _regrid_scalar_nd(self, scalar, mapxr, mapyr):
        """ Interpolates scalar onto new coordinates using given map. If known scalar is f(x,y), this function outputs f(map^{-1}(x,y))

        Args:
            scalar (np.ndarray): scalar field to regrid
            mapxr (np.ndarray): x-map for regrid in real space
            mapyr (np.ndarray): y-map for regrid in real space

        Returns:
            np.ndarray: regridded scalar
        """
        scalar_regrid = np.zeros_like(scalar)
        # Loop over intervals
        for i in range(scalar.shape[0]):
            x_mapped= mapxr[i,:,:]  - np.floor(mapxr[i,:,:]/(2*np.pi))*(2*np.pi)
            y_mapped = mapyr[i,:,:]  - np.floor(mapyr[i,:,:]/(2*np.pi))*(2*np.pi)
            interp = LinearNDInterpolator(list(zip(np.ravel(x_mapped),np.ravel(y_mapped))),np.ravel(scalar[i,:,:]))
            working_scalar_regrid = interp(self.xx, self.yy)

            # This regrid will contain some nans around x,y = 0. We shift coordinates and interpolate twice more to end up with an array without nans.
            [xx_shift1,yy_shift1,x_mapped_shift1, y_mapped_shift1]= self._shift_coord_zero([self.xx,self.yy,x_mapped,y_mapped],np.pi)
            [xx_shift2,yy_shift2,x_mapped_shift2, y_mapped_shift2]= self._shift_coord_zero([self.xx,self.yy,x_mapped,y_mapped],1.5*np.pi)
            interp_shift1 = LinearNDInterpolator(list(zip(np.ravel(x_mapped_shift1),np.ravel(y_mapped_shift1))),np.ravel(scalar[i,:,:]))
            interp_shift2 = LinearNDInterpolator(list(zip(np.ravel(x_mapped_shift2),np.ravel(y_mapped_shift2))),np.ravel(scalar[i,:,:]))

            scalar_regrid_shift1 = interp_shift1(xx_shift1, yy_shift1)
            scalar_regrid_shift2 = interp_shift2(xx_shift2, yy_shift2)

            working_scalar_regrid[np.isnan(working_scalar_regrid)] = scalar_regrid_shift1[np.isnan(working_scalar_regrid)]
            working_scalar_regrid[np.isnan(working_scalar_regrid)] = scalar_regrid_shift2[np.isnan(working_scalar_regrid)]

            scalar_regrid[i,:,:] = working_scalar_regrid

        return scalar_regrid
    
    def _regrid_scalar(self, scalar, mapxr, mapyr):
        """ Interpolates scalar onto new coordinates using given map. If known scalar is f(x,y), this function outputs f(map^{-1}(x,y))

        Args:
            scalar (np.ndarray): scalar field to regrid
            mapxr (np.ndarray): x-map for regrid in real space
            mapyr (np.ndarray): y-map for regrid in real space

        Returns:
            np.ndarray: regridded scalar
        """
        
        x_mapped= mapxr - np.floor(mapxr/(2*np.pi))*(2*np.pi)
        y_mapped = mapyr  - np.floor(mapyr/(2*np.pi))*(2*np.pi)
        interp = LinearNDInterpolator(list(zip(np.ravel(x_mapped),np.ravel(y_mapped))),np.ravel(scalar))
        working_scalar_regrid = interp(self.xx, self.yy)

        # This regrid will contain some nans around x,y = 0. We shift coordinates and interpolate twice more to end up with an array without nans.
        [xx_shift1,yy_shift1,x_mapped_shift1, y_mapped_shift1]= self._shift_coord_zero([self.xx,self.yy,x_mapped,y_mapped],np.pi)
        [xx_shift2,yy_shift2,x_mapped_shift2, y_mapped_shift2]= self._shift_coord_zero([self.xx,self.yy,x_mapped,y_mapped],1.5*np.pi)
        interp_shift1 = LinearNDInterpolator(list(zip(np.ravel(x_mapped_shift1),np.ravel(y_mapped_shift1))),np.ravel(scalar))
        interp_shift2 = LinearNDInterpolator(list(zip(np.ravel(x_mapped_shift2),np.ravel(y_mapped_shift2))),np.ravel(scalar))

        scalar_regrid_shift1 = interp_shift1(xx_shift1, yy_shift1)
        scalar_regrid_shift2 = interp_shift2(xx_shift2, yy_shift2)

        working_scalar_regrid[np.isnan(working_scalar_regrid)] = scalar_regrid_shift1[np.isnan(working_scalar_regrid)]
        working_scalar_regrid[np.isnan(working_scalar_regrid)] = scalar_regrid_shift2[np.isnan(working_scalar_regrid)]

        scalar_regrid = working_scalar_regrid

        return scalar_regrid
    
    def _regrid_scalar_inverse_nd(self, scalar,mapxr,mapyr):
        """ Interpolates scalar onto new coordinates using inverse of given map. If known scalar is f(x,y), this function outputs f(map(x,y))

        Args:
            scalar (np.ndarray): scalar field to regrid
            mapxr (np.ndarray): x-map for regrid in real space
            mapyr (np.ndarray): y-map for regrid in real space

        Returns:
            np.ndarray: regridded scalar
        """
        scalar_interp = np.zeros_like(scalar)
        for i in range(scalar.shape[0]):
            # Maps from mapped coordinate to original - i.e. mapmid will map from instantaneous midpoint scalar to phi(T)
            x_mapped = mapxr[i,:,:]  - np.floor(mapxr[i,:,:]/(2*np.pi))*(2*np.pi)
            y_mapped = mapyr[i,:,:]  - np.floor(mapyr[i,:,:]/(2*np.pi))*(2*np.pi)
            scalar_pad = self._pad(scalar[i,:,:])
            interp_f = RectBivariateSpline(self.y_padd, self.x_padd, scalar_pad, kx=3, ky=3)
            scalar_interp_ravel = interp_f.ev(np.ravel(y_mapped), np.ravel(x_mapped))
            scalar_interp[i,:,:] = np.reshape(scalar_interp_ravel, (scalar_interp.shape[1], scalar_interp.shape[2]))
        return scalar_interp

    def _shift_coord_zero(self,in_coord_list,shift):
        """Shifts the zero of any coordinate by distance 'shift'

        Args:
            in_coord_list (list(np.ndarray)): list of 2D coordinate arrays to shift
            shift (float): Distance by which to shift coordinate

        Returns:
            list(np.ndarray)): list of shifted 2D coordinate arrays
        """
        out_coord_list = []
        for in_coord in in_coord_list:
            out_coord = np.copy(in_coord)
            out_coord[in_coord < shift] +=(2*np.pi - shift)
            out_coord[in_coord >= shift] -= shift
            out_coord_list.append(out_coord)
        return out_coord_list

    def _interp(self,scalar,X_in,Y_in):
        """ Interpolates to find scalar field at mapped coordinates, then outputs this array in real and spectral space. if scalar is f, 
        this finds the Fourier transform of f*Xi (composed)

        Args:
            scalar (np.ndarray): scalar to be interpolated in real space
            X_in (np.ndarray): mapped x-coordinate
            Y_in (np.ndarray): mapped y-coordinate

        Returns:
            np.ndarray: interpolated scalar in real space
            np.ndarray: interpolated scalar in spectral space
        """
        scalar_pad = self._pad(scalar)
        interp_f = RectBivariateSpline(self.y_padd, self.x_padd, scalar_pad, kx=3, ky=3)
        scalar_interp = interp_f.ev(np.ravel(Y_in), np.ravel(X_in))
        scalar_interp = np.reshape(scalar_interp, (scalar.shape[0], scalar.shape[1]))
        return scalar_interp, rfft2(scalar_interp)
    
    def _pad(self,in_field):
        """ Pads a (periodic) array with one extra grid-cell at end boundary

        Args:
            in_field (np.ndarray): field to be padded
        Returns:
            np.ndarray: padded field
        """
        Q = np.zeros((in_field.shape[0]+1,in_field.shape[1]+1))
        Q[0:-1,0:-1] = in_field
        Q[-1,:] = Q[0,:]
        Q[:,-1] = Q[:,0]
        
        return Q
    
    def _make_dataset_LM(self):
        """ Make an xarray dataset from final Lagrangian mean data

        Returns:
            xarray.core.dataset.Dataset: xarray dataset with permutations of midpoint scalars and Lagrangian means
        """
        data_vars = dict(
                    z_inst=(["t","y", "x"], self.z_inst),
                    z_EM=(["t","y", "x"], self.z_EMr),
                    timestep = (["t"],np.array(self.timesteps_LM))
        )
        if self.strat ==1:
            data_vars['z_LM_at_end'] = (["t","y", "x"],self.z_LMr_at_end)
            if self.solve_xi12:
                data_vars['z_LM_at_mean'] = (["t","y", "x"], self.z_LMr_at_mean)
                data_vars['Xi12_x'] = (["t","y", "x"], irfft2(self.xi12xk) + self.xx)
                data_vars['Xi12_y'] = (["t","y", "x"], irfft2(self.xi12yk) + self.yy)
            if self.solve_xi13:
                data_vars['z_LM_at_mid'] = (["t","y", "x"], self.z_LMr_at_mid)
                data_vars['Xi13_x'] = (["t","y", "x"], irfft2(self.xi13xk) + self.xx)
                data_vars['Xi13_y'] = (["t","y", "x"], irfft2(self.xi13yk) + self.yy)
                data_vars['z_inst_at_end'] = (["t","y", "x"], self.z_inst_at_end)
            if self.solve_xi13 and self.solve_xi12:
                data_vars['z_inst_at_mean'] = (["t","y", "x"], self.z_inst_at_mean)
        elif self.strat ==2:
            data_vars['z_LM_at_mean'] = (["t","y", "x"],self.z_LMr_at_mean)
            data_vars['z_LM_at_end'] = (["t","y", "x"], self.z_LMr_at_end)
            data_vars['Xi21_x'] = (["t","y", "x"], irfft2(self.xi21xk) + self.xx)
            data_vars['Xi21_y'] = (["t","y", "x"], irfft2(self.xi21yk) + self.yy)
            if self.solve_xi23:
                data_vars['z_LM_at_mid'] = (["t","y", "x"],self.z_LMr_at_mid)
                data_vars['Xi23_x'] = (["t","y", "x"], irfft2(self.xi23xk) + self.xx)
                data_vars['Xi23_y'] = (["t","y", "x"], irfft2(self.xi23yk) + self.yy)
                data_vars['z_inst_at_mean'] = (["t","y", "x"], self.z_inst_at_mean)
                data_vars['z_inst_at_end'] = (["t","y", "x"], self.z_inst_at_end)
        elif self.strat==3:
            data_vars['z_inst_at_end'] = (["t","y", "x"], self.z_inst_at_end)
            data_vars['z_LM_at_mid'] = (["t","y", "x"],self.z_LMr_at_mid)
            data_vars['z_LM_at_end'] = (["t","y", "x"], self.z_LMr_at_end)
            data_vars['Xi31_x'] = (["t","y", "x"], irfft2(self.xi31xk) + self.xx)
            data_vars['Xi31_y'] = (["t","y", "x"], irfft2(self.xi31yk) + self.yy)
            if self.solve_xi32:
                data_vars['z_inst_at_mean'] = (["t","y", "x"], self.z_inst_at_mean)
                data_vars['z_LM_at_mean'] = (["t","y", "x"], self.z_LMr_at_mean)
                data_vars['Xi32_x'] = (["t","y", "x"], irfft2(self.xi32xk) + self.xx)
                data_vars['Xi32_y'] = (["t","y", "x"], irfft2(self.xi32yk) + self.yy)
            
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                x=(["x"], self.x),
                y=(["y"], self.y),
                t=(["t"], np.linspace(self.interval_start[0] + self.Ntinthf*self.dt,self.interval_start[0] + self.Ntinthf*self.dt+ self.Nint*self.dtslow,self.Nint,endpoint=False)),
            ),
            attrs=dict(description="SW GLM solver outputs")
        )
        ds.z_inst.attrs["long_name"] = "Instantaneous midpoint vorticity"
        ds.z_EM.attrs["long_name"] = "(Weighted) Eulerian mean vorticity"
        ds.x.attrs["long_name"] = "x"
        ds.y.attrs["long_name"] = "y"
        ds.y.attrs["long_name"] = "t"
        
        
        if self.strat == 1:
            ds.z_LM_at_end.attrs["long_name"] = "GLM vorticity at endpoint"
            if self.solve_xi12:
                ds.z_LM_at_mean.attrs["long_name"] = "GLM vorticity at mean"
                ds.Xi12_x.attrs["long_name"] = "map from endpoint to mean coords (x)"
                ds.Xi12_y.attrs["long_name"] = "map from endpoint to mean coords (y)"
            if self.solve_xi13:
                ds.z_LM_at_mid.attrs["long_name"] = "GLM vorticity at midpoint"
                ds.Xi13_x.attrs["long_name"] = "map from endpoint to midpoint coords (x)"
                ds.Xi13_y.attrs["long_name"] = "map from endpoint to midpoint coords (y)"
                ds.z_inst_at_end.attrs["long_name"] = "Instantaneous midpoint vorticity, remapped to endpoint coords"
            if self.solve_xi13 and self.solve_xi12:
                ds.z_inst_at_mean.attrs["long_name"] = "Instantaneous midpoint vorticity, remapped to mean coords"
            
        elif self.strat ==2:
            ds.z_LM_at_mean.attrs["long_name"] = "GLM vorticity at mean"
            ds.z_LM_at_end.attrs["long_name"] = "GLM vorticity at endpoint"
            ds.Xi21_x.attrs["long_name"] = "map from mean to endpoint coords (x)"
            ds.Xi21_y.attrs["long_name"] = "map from mean to endpoint coords (y)"
            if self.solve_xi23: 
                ds.Xi23_x.attrs["long_name"] = "map from mean to midpoint coords (x)"
                ds.Xi23_y.attrs["long_name"] = "map from mean to midpoint coords (y)"
                ds.z_inst_at_mean.attrs["long_name"] = "Instantaneous midpoint vorticity, remapped to mean coords"
                ds.z_LM_at_mid.attrs["long_name"] = "GLM vorticity at midpoint"
                ds.z_inst_at_end.attrs["long_name"] = "Instantaneous midpoint vorticity, remapped to endpoint coords"
        elif self.strat==3:
            ds.z_inst_at_end.attrs["long_name"] = "Instantaneous midpoint vorticity, remapped to endpoint coords"
            ds.z_LM_at_mid.attrs["long_name"] = "GLM vorticity at midpoint"
            ds.Xi31_x.attrs["long_name"] = "map from midpoint to endpoint coords (x)"
            ds.Xi31_y.attrs["long_name"] = "map from midpoint to endpoint coords (y)"
            ds.z_LM_at_end.attrs["long_name"] = "GLM vorticity at endpoint"
            if self.solve_xi32:
                ds.z_inst_at_mean.attrs["long_name"] = "Instantaneous midpoint vorticity, remapped to mean coords"
                ds.z_LM_at_mean.attrs["long_name"] = "GLM vorticity at mean"
                ds.Xi32_x.attrs["long_name"] = "map from midpoint to mean coords (x)"
                ds.Xi32_y.attrs["long_name"] = "map from midpoint to mean coords (y)"  
                
        if self.Nint ==1:
            ds = ds.isel({'t':0})
        return ds

    def _make_dataset_LM_velocity(self):
        """
        Make an xarray dataset from final Lagrangian-mean VELOCITY data
        """

        data_vars = dict(
            u_inst=(["t", "z", "y", "x"], self.u_inst),
            v_inst=(["t", "z", "y", "x"], self.v_inst),
            w_inst=(["t", "z", "y", "x"], self.w_inst),

            u_EM=(["t", "z", "y", "x"], self.u_EMr),
            v_EM=(["t", "z", "y", "x"], self.v_EMr),
            w_EM=(["t", "z", "y", "x"], self.w_EMr),

            timestep=(["t"], np.array(self.timesteps_LM)),
        )

        # ---- STRATEGY 1 (endpoint reference) ----
        if self.strat == 1:
            data_vars.update({
                "u_LM_at_end": (["t", "z", "y", "x"], self.u_LMr_at_end),
                "v_LM_at_end": (["t", "z", "y", "x"], self.v_LMr_at_end),
                "w_LM_at_end": (["t", "z", "y", "x"], self.w_LMr_at_end),
            })

            if self.solve_xi12:
                data_vars.update({
                    "u_LM_at_mean": (["t", "z", "y", "x"], self.u_LMr_at_mean),
                    "v_LM_at_mean": (["t", "z", "y", "x"], self.v_LMr_at_mean),
                    "w_LM_at_mean": (["t", "z", "y", "x"], self.w_LMr_at_mean),

                    "Xi12_x": (["t", "y", "x"], irfft2(self.xi12xk) + self.xx),
                    "Xi12_y": (["t", "y", "x"], irfft2(self.xi12yk) + self.yy),
                })

            if self.solve_xi13:
                data_vars.update({
                    "u_LM_at_mid": (["t", "z", "y", "x"], self.u_LMr_at_mid),
                    "v_LM_at_mid": (["t", "z", "y", "x"], self.v_LMr_at_mid),
                    "w_LM_at_mid": (["t", "z", "y", "x"], self.w_LMr_at_mid),

                    "Xi13_x": (["t", "y", "x"], irfft2(self.xi13xk) + self.xx),
                    "Xi13_y": (["t", "y", "x"], irfft2(self.xi13yk) + self.yy),
                })

        # ---- Build dataset ----
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                x=(["x"], self.x),
                y=(["y"], self.y),
                z=(["z"], self.z),
                t=(["t"], np.linspace(
                    self.interval_start[0] + self.Ntinthf * self.dt,
                    self.interval_start[0] + self.Ntinthf * self.dt + self.Nint * self.dtslow,
                    self.Nint,
                    endpoint=False,
                )),
            ),
            attrs=dict(description="3D GLM velocity output"),
        )

        # ---- Metadata ----
        ds.u_inst.attrs["long_name"] = "Instantaneous velocity (x)"
        ds.v_inst.attrs["long_name"] = "Instantaneous velocity (y)"
        ds.w_inst.attrs["long_name"] = "Instantaneous velocity (z)"

        ds.u_EM.attrs["long_name"] = "Eulerian-mean velocity (x)"
        ds.v_EM.attrs["long_name"] = "Eulerian-mean velocity (y)"
        ds.w_EM.attrs["long_name"] = "Eulerian-mean velocity (z)"

        if self.Nint == 1:
            ds = ds.isel(t=0)

        return ds

    def _make_dataset_inst(self):
        """ Make an xarray dataset from instantaneous final data, at y value self.ny_inst

        Returns:
            xarray.core.dataset.Dataset: xarray dataset with permutations of midpoint scalars and Lagrangian means
        """
    
        if self.solve_LM and self.kernel_type =='exponential':
            data_vars = dict(
                z_ts=(["t","x"], self.z_ts),
                u_ts=(["t","x"], self.u_ts),
                v_ts=(["t","x"], self.v_ts),
                z_LM_ts=(["t","x"], self.z_LM_ts),
                z_EM_ts=(["t","x"], self.z_EM_ts),
                timestep = (["t"],np.array(self.timesteps_inst))
            )
            
        else:
            data_vars = dict(
                z_ts=(["t","x"], self.z_ts),
                u_ts=(["t","x"], self.u_ts),
                v_ts=(["t","x"], self.v_ts),
                timestep = (["t"],np.array(self.timesteps_inst))
            ) 
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                x=(["x"], self.x),
                t=(["t"], np.array(self.timesteps_inst)*self.dt),
            ),
            attrs=dict(description="SW solver instantaneous outputs")
        )
        ds.z_ts.attrs["long_name"] = f"Instantaneous vorticity at y index {self.ny_inst}"
        ds.u_ts.attrs["long_name"] = f"Instantaneous u at y index {self.ny_inst}"
        ds.v_ts.attrs["long_name"] = f"Instantaneous v aty index {self.ny_inst}"
        if self.solve_LM and self.kernel_type =='exponential':
            ds.z_LM_ts.attrs["long_name"] = f"Exponential LM vorticity at y index {self.ny_inst}"
            ds.z_EM_ts.attrs["long_name"] = f"Exponential EM vorticity at y index {self.ny_inst}"
        
        if self.Nint ==1:
            ds = ds.isel({'t':0})
        return ds

    def plot_vorticity(self,iTime=0):
        """ Plotting function to plot shallow water vorticity

        Args:
            iTime (int, optional): timestep. Defaults to 0.
        """
        fig, ax = plt.subplots(1,1,figsize = (5,4),constrained_layout=True)
        p0 = ax.pcolormesh(self.zr, cmap='RdBu_r',vmin = -1, vmax = 1)
        
        fig.colorbar(p0,ax=ax,label='Vorticity')
       
        ax.axes.set_xticklabels([]) 
        ax.axes.set_yticklabels([]) 
        fig.savefig(self.movie_dir+'movie_still_'+f'{(int(iTime/self.movie_freq)):04d}'+'.png')
       
        plt.close()
    
    def plot_u(self,iTime=0):
        """ Plotting function to plot shallow water x-velocity

        Args:
            iTime (int, optional): timestep. Defaults to 0.
        """
        fig, ax = plt.subplots(1,1,figsize = (5,4),constrained_layout=True)
        p0 = ax.pcolormesh(self.ur, cmap='RdBu_r',vmin = -0.1, vmax=0.1)
         
        fig.colorbar(p0,ax=ax,label='Velocity (u)')
       
        ax.axes.set_xticklabels([]) 
        ax.axes.set_yticklabels([]) 
        fig.savefig(self.movie_dir+'movie_still_'+f'{(int(iTime/self.movie_freq)):04d}'+'.png')
       
        plt.close()

    def plot_pv(self,iTime=0):
        """ Plotting function to plot shallow water potential vorticity perturbation

        Args:
            iTime (int, optional): timestep. Defaults to 0.
        """
        fig, ax = plt.subplots(1,1,figsize = (5,4),constrained_layout=True)
        p0 = ax.pcolormesh((self.zr + 1/self.Ro)/self.hr -1/self.Ro , cmap='RdBu_r')
        
        fig.colorbar(p0,ax=ax,label='PV perturbation')
       
        ax.axes.set_xticklabels([]) 
        ax.axes.set_yticklabels([]) 
        fig.savefig(self.movie_dir+'movie_still_'+f'{(int(iTime/self.movie_freq)):04d}'+'.png')
       
        plt.close()

    def plot_v(self,iTime=0):
        """ Plotting function to plot shallow water y-velocity

        Args:
            iTime (int, optional): timestep. Defaults to 0.
        """
        fig, ax = plt.subplots(1,1,figsize = (5,4),constrained_layout=True)
        p0 = ax.pcolormesh(self.vr, cmap='RdBu_r')
        
        fig.colorbar(p0,ax=ax,label='Velocity (v)')
       
        ax.axes.set_xticklabels([]) 
        ax.axes.set_yticklabels([]) 
        fig.savefig(self.movie_dir+'movie_still_'+f'{(int(iTime/self.movie_freq)):04d}'+'.png')
       
        plt.close()

    def plot_jacobian(self,iTime=0):
        """ Plotting function to jacobian of Xi

        Args:
            iTime (int, optional): timestep. Defaults to 0.
        """
        # First calculate it:
      
        xi21x_xr = irfft2(1j * self.Kxx * self.xi21xk[0,:,:])
        xi21x_yr = irfft2(1j * self.Kyy * self.xi21xk[0,:,:])
        xi21y_xr = irfft2(1j * self.Kxx * self.xi21yk[0,:,:])
        xi21y_yr = irfft2(1j * self.Kyy * self.xi21yk[0,:,:])
        J = (1 + xi21x_xr)*(1+ xi21y_yr) - xi21x_yr*xi21y_xr
        
        fig, ax = plt.subplots(1,1,figsize = (5,4),constrained_layout=True)
        p0 = ax.pcolormesh(J, cmap='RdBu_r',vmin = -4,vmax = 6)
        
        fig.colorbar(p0,ax=ax,label='Jacobian')
       
        ax.axes.set_xticklabels([]) 
        ax.axes.set_yticklabels([]) 
        fig.savefig(self.movie_dir+'movie_still_'+f'{(int(iTime/self.movie_freq)):04d}'+'.png')
       
        plt.close()

    def plot_vorticity_means(self,iTime=0):
        """ Plotting function to plot evolution of vorticity means

        Args:
            iTime (int, optional): timestep. Defaults to 0.
        """
        self.z_LMr = irfft2(self.z_LMk)
        fig, ax = plt.subplots(1,3,figsize = (13,4),constrained_layout=True)
        p0 = ax[0].pcolormesh(self.zr, cmap='RdBu_r',vmin = -1,vmax = 1)
        p1 = ax[1].pcolormesh(self.z_LMr[0,:,:], cmap='RdBu_r',vmin = -1,vmax = 1)
        p2 = ax[2].pcolormesh(self.z_EMr[0,:,:], cmap='RdBu_r',vmin = -1,vmax = 1)
        
        fig.colorbar(p0,ax=ax[2],label=r'Vorticity')
       
        ax[0].set_title('Vorticity')
        ax[1].set_title('Partial Lagrangian mean vorticity')
        ax[2].set_title('Partial Eulerian mean vorticity')
        [ax[i].axes.set_xticklabels([]) for i in range(3)];
        [ax[i].axes.set_yticklabels([]) for i in range(3)];
        fig.savefig(self.movie_dir+'movie_still_'+f'{(int(iTime/self.movie_freq)):04d}'+'.png')
        
        plt.close()


    def plot_vars(self,filename='output_vars.png',vmin = [None for i in range(5)],vmax = [None for i in range(5)]):
        """ Plotting function to show midpoint and Lagrangian/Eulerian mean of scalars

        Args:
            filename (str, optional): file name for figure. Defaults to 'figure.png'.
            vmin (list, optional): colorbar minimum for each panel. Defaults to [None for i in range(5)].
            vmax (list, optional): colorbar maximum for each panel. Defaults to [None for i in range(5)].
        """
       
        fig, ax = plt.subplots(1,5,figsize = (20,3),constrained_layout=True)
        p0 = ax[0].pcolormesh(self.zr,vmin=vmin[0], vmax=vmax[0],cmap='RdBu_r')
        p1 = ax[1].pcolormesh(self.hr -1,vmin=vmin[0], vmax=vmax[0],cmap='RdBu_r')
        p2 = ax[1].pcolormesh((self.zr + 1/self.Ro)/self.hr - 1/self.Ro ,vmin=vmin[0], vmax=vmax[0],cmap='RdBu_r')
        p3 = ax[3].pcolormesh(self.ur,vmin=vmin[0], vmax=vmax[0],cmap='RdBu_r')
        p4 = ax[4].pcolormesh(self.vr,vmin=vmin[0], vmax=vmax[0],cmap='RdBu_r')
        
        fig.colorbar(p0,ax=ax[0],label='Vorticity')
        fig.colorbar(p1,ax=ax[1],label='Height perturbation')
        fig.colorbar(p2,ax=ax[2],label='PV perturbation')
        fig.colorbar(p3,ax=ax[2],label='u')
        fig.colorbar(p4,ax=ax[2],label='v')

        [ax[i].axes.set_xticklabels([]) for i in range(5)];
        [ax[i].axes.set_yticklabels([]) for i in range(5)];
        [ax[i].axes.set_xlabel('') for i in range(5)]
        [ax[i].axes.set_ylabel('') for i in range(5)];
        plt.show()
        fig.savefig(filename)

    def plot_comparison(self,interval=0,filename='comparison.png',vmin = [None for i in range(3)], vmax = [None for i in range(3)]):
        """ Plotting function for mean/wave decomposition according to different wave definitions

        Args:
            filename (str, optional): file name to save figure to. Defaults to 'comparison.png'.
            vmin (list, optional): colorbar minimum for each column. Defaults to [None for i in range(3)].
            vmax (list, optional): colorbar maximum for each column. Defaults to [None for i in range(3)].
        """

        ds = self.ds_LM.isel({'t':interval})

        fig, ax = plt.subplots(3,3,figsize = (10,9), constrained_layout=True)
        ax[0,0].pcolormesh(ds.z_inst, vmin = vmin[0], vmax = vmax[0], cmap = 'RdBu_r')
        ax[0,1].pcolormesh(ds.z_LM_at_mean, vmin = vmin[1], vmax = vmax[1], cmap = 'RdBu_r')
        ax[0,2].pcolormesh(ds.z_inst - ds.z_LM_at_mean, vmin = vmin[2], vmax = vmax[2], cmap = 'RdBu_r')
        ax[0,0].text(0.08,0.9,r'$f(\mathbf{x},t^*)$',fontsize=16,transform=ax[0,0].transAxes)
        ax[0,1].text(0.08,0.9,r'$\bar f(\mathbf{x},t^*)$',fontsize=16,transform=ax[0,1].transAxes)
        ax[0,2].text(0.08,0.9,r'$f(\mathbf{x},t^*) - \bar f(\mathbf{x},t^*)$',fontsize=16,transform=ax[0,2].transAxes)

        ax[1,0].pcolormesh(ds.z_inst, vmin = vmin[0], vmax = vmax[0], cmap = 'RdBu_r')
        ax[1,1].pcolormesh(ds.z_LM_at_mid, vmin = vmin[1], vmax = vmax[1], cmap = 'RdBu_r')
        ax[1,2].pcolormesh(ds.z_inst - ds.z_LM_at_mid, vmin = vmin[2], vmax = vmax[2], cmap = 'RdBu_r')
        ax[1,0].text(0.08,0.9,r'$f(\mathbf{x},t^*)$',fontsize=16,transform=ax[1,0].transAxes)
        ax[1,1].text(0.08,0.9,r'$\bar f(\boldsymbol{\Xi}^{-1}(\mathbf{x},t^*),t^*)$',fontsize=16,transform=ax[1,1].transAxes)
        ax[1,2].text(0.08,0.9,r'$f(\mathbf{x},t^*) - \bar f(\boldsymbol{\Xi}^{-1}(\mathbf{x},t^*),t^*)$',fontsize=16,transform=ax[1,2].transAxes)

        ax[2,0].pcolormesh(ds.z_inst_at_mean, vmin = vmin[0], vmax = vmax[0], cmap = 'RdBu_r')
        ax[2,1].pcolormesh(ds.z_LM_at_mean, vmin = vmin[1], vmax = vmax[1], cmap = 'RdBu_r')
        ax[2,2].pcolormesh(ds.z_inst_at_mean - ds.z_LM_at_mean, vmin = vmin[2], vmax = vmax[2], cmap = 'RdBu_r')
        ax[2,0].text(0.08,0.9,r'$f(\boldsymbol{\Xi}(\mathbf{x},t^*),t^*)$',fontsize=16,transform=ax[2,0].transAxes)
        ax[2,1].text(0.08,0.9,r'$\bar f(\mathbf{x},t^*)$',fontsize=16,transform=ax[2,1].transAxes)
        ax[2,2].text(0.08,0.9,r'$f(\boldsymbol{\Xi}(\mathbf{x},t^*),t^*) - \bar f(\mathbf{x},t^*)$',fontsize=16,transform=ax[2,2].transAxes)


        [ax[i,j].axes.set_xticklabels([]) for i in range(3) for j in range(3)];
        [ax[i,j].axes.set_yticklabels([]) for i in range(3) for j in range(3)];
    
        fig.savefig(filename)

    def plot_energy_spectrum(self,filename='spectrum.png'):
        Nt = int(np.ceil(self.Ttotal/self.dt)+1)
        omega_sig = (np.fft.fftfreq(Nt)*2*np.pi/self.dt)[:int(Nt/2)]
        sig = ((np.abs(np.fft.fft(self.u_ts[:,int(self.Nx/2)],axis=0))**2 + np.abs(np.fft.fft(self.v_ts[:,int(self.Nx/2)],axis=0))**2)/Nt**2)[:int(Nt/2)]
        fig, ax = plt.subplots(1,1,figsize = (10,5))
        ax.plot(omega_sig,sig,'r',label=r'$E(\omega)$')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.plot(np.array(self.wave_omega), self.E/2/(np.array(self.wave_omega))**2,'k',label=r'$E(\omega) \sim \omega^{-2}$')
        ax.plot(np.array(self.wave_omega), self.wave_E,'kx',label='Initialised freqencies')
        ax.legend()
        ax.set_xlim([1,1000])
        ax.set_xlabel(r'Frequency $\omega$ [rad/s]')
        ax.set_ylabel(r'Spectral kinetic energy $E(\omega)$')
        plt.show()
        fig.savefig(filename)

    def save(self,filename):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)
        file_stats = os.stat(filename)
        print(f'Saved object to {filename}, size {file_stats.st_size / (1024 * 1024):.2f}MB')


    def bk_solve(self):
        """ Main routine to run solver
        """
        t_start = time.time()
        self.timestep = 0
        self.timesteps_LM = []
        self.timesteps_inst = [0]

        # Initialise instantaneous fields to save
        # At interval frequency
        self.z_inst = np.zeros((self.Nint, self.Ny, self.Nx))
        self.zr = irfft2(self.zk)
        if self.save_inst:
            # At high frequency, let's just save a slice through the middle
            self.z_ts = np.zeros((int(np.floor(self.Nttotal / self.Nt_save_inst)) + 1, self.Nx))
            self.u_ts = np.zeros((int(np.floor(self.Nttotal / self.Nt_save_inst)) + 1, self.Nx))
            self.v_ts = np.zeros((int(np.floor(self.Nttotal / self.Nt_save_inst)) + 1, self.Nx))
            self.z_ts[0, :] = self.zr[self.ny_inst, :]
            self.u_ts[0, :] = self.ur[self.ny_inst, :]
            self.v_ts[0, :] = self.vr[self.ny_inst, :]
            if self.solve_LM and self.kernel_type == 'exponential':
                self.z_LM_ts = np.zeros((int(np.floor(self.Nttotal / self.Nt_save_inst)) + 1, self.Nx))
                self.z_EM_ts = np.zeros((int(np.floor(self.Nttotal / self.Nt_save_inst)) + 1, self.Nx))
        # First, we'll run until we should start solving LM equations
        self._do_timesteps(start_step=0, end_step=self.LM_start_step)

        # Then, we'll run until time t_LM_start + T/2:
        self._do_timesteps(self.LM_start_step, end_step=self.LM_start_step + self.Ntinthf)
        self.z_inst[0, :, :] = self.zr
        self.timesteps_LM.append(self.timestep)

        # Then run saving instantaneous every dt_slow
        if self.Nint > 1:
            for interval in range(1, self.Nint):
                self._do_timesteps(start_step=self.LM_start_step + self.Ntinthf + (interval - 1) * self.Ntgap,
                                   end_step=self.LM_start_step + self.Ntinthf + interval * self.Ntgap)
                self.z_inst[interval, :, :] = self.zr
                self.timesteps_LM.append(self.timestep)

        # Finish running to end
        self._do_timesteps(start_step=self.LM_start_step + self.Ntinthf + (self.Nint - 1) * self.Ntgap,
                           end_step=self.LM_start_step + self.Ntint + (self.Nint - 1) * self.Ntgap)

        t_end_sim = time.time()
        print('computation time', t_end_sim - t_start)

        if self.solve_LM:
            # Define Lagrangian mean fields
            self.z_LMr = irfft2(self.z_LMk)
            # Do regridding
            if self.strat == 1:
                self.z_LMr_at_end = self.z_LMr
                if self.solve_xi12:
                    Xi12xr = irfft2(self.xi12xk) + self.xx
                    Xi12yr = irfft2(self.xi12yk) + self.yy
                    self.z_LMr_at_mean = self._regrid_scalar_nd(self.z_LMr, Xi12xr, Xi12yr)
                if self.solve_xi13:
                    Xi13xr = irfft2(self.xi13xk) + self.xx
                    Xi13yr = irfft2(self.xi13yk) + self.yy
                    self.z_LMr_at_mid = self._regrid_scalar_nd(self.z_LMr, Xi13xr, Xi13yr)
                    # Regrid instantaneous fields to endpoint coordinate
                    self.z_inst_at_end = self._regrid_scalar_inverse_nd(self.z_inst, Xi13xr, Xi13yr)
                if self.solve_xi13 and self.solve_xi12:
                    # Regrid instantaneous fields from endpoint to mean coordinate
                    self.z_inst_at_mean = self._regrid_scalar_nd(self.z_inst_at_end, Xi12xr, Xi12yr)

            elif self.strat == 2:
                Xi21xr = irfft2(self.xi21xk) + self.xx
                Xi21yr = irfft2(self.xi21yk) + self.yy
                self.z_LMr_at_mean = self.z_LMr
                self.z_LMr_at_end = self._regrid_scalar_nd(self.z_LMr, Xi21xr, Xi21yr)
                if self.solve_xi23:
                    Xi23xr = irfft2(self.xi23xk) + self.xx
                    Xi23yr = irfft2(self.xi23yk) + self.yy
                    self.z_LMr_at_mid = self._regrid_scalar_nd(self.z_LMr, Xi23xr, Xi23yr)
                    # Regrid instantaneous fields from endpoint to mean and end coordinate
                    self.z_inst_at_mean = self._regrid_scalar_inverse_nd(self.z_inst, Xi23xr, Xi23yr)
                    self.z_inst_at_end = self._regrid_scalar_nd(self.z_inst_at_mean, Xi21xr, Xi21yr)

            elif self.strat == 3:
                self.z_LMr_at_mid = self.z_LMr
                Xi31xr = irfft2(self.xi31xk) + self.xx
                Xi31yr = irfft2(self.xi31yk) + self.yy
                self.z_LMr_at_end = self._regrid_scalar_nd(self.z_LMr, Xi31xr, Xi31yr)
                # Regrid instantaneous fields to endpoint coordinate
                self.z_inst_at_end = self._regrid_scalar_nd(self.z_inst, Xi31xr, Xi31yr)
                if self.solve_xi32:
                    Xi32xr = irfft2(self.xi32xk) + self.xx
                    Xi32yr = irfft2(self.xi32yk) + self.yy
                    self.z_LMr_at_mean = self._regrid_scalar_nd(self.z_LMr, Xi32xr, Xi32yr)
                    # Regrid instantaneous fields from mid to mean coordinate
                    self.z_inst_at_mean = self._regrid_scalar_nd(self.z_inst, Xi32xr, Xi32yr)
        t_end_regrid = time.time()
        print('regridding time', t_end_regrid - t_end_sim)
        print('total time', t_end_regrid - t_start)

        # Make an output xarray dataset to store outputs and metadata
        if self.solve_LM:
            self.ds_LM = self._make_dataset_LM()
        if self.save_inst:
            self.ds_inst = self._make_dataset_inst()

        # Make a movie
        if self.movie_type is not None:
            i = 2

            while os.path.exists(f'{self.movie_dir + self.movie_fname}.mp4'):
                self.movie_fname = self.movie_fname.split('_')[0] + f'_{i}'

                i += 1
            os.system(
                f'ffmpeg -framerate 20 -start_number 0 -i ' + self.movie_dir + f'movie_still_%04d.png -pix_fmt yuv420p {self.movie_dir + self.movie_fname}.mp4')
            os.system(f'rm ' + self.movie_dir + 'movie_still*.png')

        # Save the solver object
        if self.solve_LM and self.save_solver:
            if (self.kernel_type == 'lowpass') or (self.kernel_type == 'Butterworth') or (
                    self.kernel_type == 'lowbandpass'):
                self.solver_fname = self.save_dir + f"solver_{self.kernel_type}_Nx_{self.Nx}_omega_{self.kernel_params['omega_crit']}_Nint_{self.Nint}_strat_{self.strat}_T_{self.T:.1f}_Ttotal_{self.Ttotal:.1f}.pkl"
            else:
                self.solver_fname = self.save_dir + f'solver_{self.kernel_type}_Nx_{self.Nx}_Nint_{self.Nint}_strat_{self.strat}_T_{self.T:.1f}_Ttotal_{self.Ttotal:.1f}.pkl'
            #  Could append rather than overwrite:
            # i = 2
            # while os.path.exists(f'{self.solver_fname}'):
            #     self.solver_fname = self.solver_fname.split('-')[0] + f'-{i}'
            #     i+=1
            self.save(self.solver_fname)
        elif (self.solve_LM == False) and self.save_solver:
            self.solver_fname = self.save_dir + f'solver_noLM_Nx_{self.Nx}_Nint_{self.Nint}_T_{self.T:.1f}_Ttotal_{self.Ttotal:.1f}.pkl'
            self.save(self.solver_fname)


    def load_solver(filename):
        """Load pickled solver

        Args:
            filename (str): pre-saved solver object filename

        Returns:
            SWSolver: solver object
        """
        filehandler = open(filename, 'rb')
        file_stats = os.stat(filename)
        print(f'Loaded object from {filename} to memory, size is {file_stats.st_size / (1024 * 1024):.2f}MB')
        return pickle.load(filehandler)


    def _update_LM_fields_from_velocity(self, ur, vr):
        """
        Update Lagrangian-mean fields for velocities U and V using the ξ-maps.
        Works for all intervals and kernel types.

        Args:
            ur (np.ndarray): instantaneous U velocity (Ny, Nx)
            vr (np.ndarray): instantaneous V velocity (Ny, Nx)
        """
        # Transform velocities to spectral space
        uk = rfft2(ur)
        vk = rfft2(vr)

        # Initialize LM arrays if not already done
        if not hasattr(self, 'U_LMk'):
            self.U_LMk = np.zeros((self.Nint, self.Ny, int(self.Nx/2+1)), dtype=complex)
            self.V_LMk = np.zeros((self.Nint, self.Ny, int(self.Nx/2+1)), dtype=complex)
            self.U_EMr = np.zeros((self.Nint, self.Ny, self.Nx))
            self.V_EMr = np.zeros((self.Nint, self.Ny, self.Nx))

        for interval in range(self.Nint):
            # --- Lagrangian mean update in spectral space ---
            if self.kernel_type == 'exponential':
                alpha_dt = self.alpha * self.dt
                self.U_LMk[interval,:,:] = (1 - alpha_dt) * self.U_LMk[interval,:,:] + alpha_dt * uk
                self.V_LMk[interval,:,:] = (1 - alpha_dt) * self.V_LMk[interval,:,:] + alpha_dt * vk
            else:
                self.U_LMk[interval,:,:] += uk * self.kernel_current[interval] * self.dt
                self.V_LMk[interval,:,:] += vk * self.kernel_current[interval] * self.dt

            # --- Eulerian mean update ---
            if self.kernel_type != 'exponential':
                self.U_EMr[interval,:,:] += ur * self.kernel_current[interval] * self.dt
                self.V_EMr[interval,:,:] += vr * self.kernel_current[interval] * self.dt

        # Convert spectral LM fields back to real space for storage / plotting
        self.U_LMr = irfft2(self.U_LMk)
        self.V_LMr = irfft2(self.V_LMk)



    def solve_offline_from_velocity(self, vel_ds):
        """
        Offline GLM solver using prescribed xarray velocity fields.

        Args:
            vel_ds (xarray.Dataset): Dataset containing 'UVEL' and 'VVEL' with dimensions (time, y, x)
        """
        import time
        t_start = time.time()

        Nt_total = vel_ds.dims['time']
        self.Nt_total = Nt_total

        # Initialise LM arrays and maps
        self.set_LM_IC(kernel_type=self.kernel_type, kernel_params=self.kernel_params,
                       solve_xi12=self.solve_xi12, solve_xi23=self.solve_xi23,
                       solve_xi13=self.solve_xi13, solve_xi32=self.solve_xi32)

        # Prepare timesteps for Lagrangian accumulation
        self.timesteps_LM = []

        # Initialise flag array
        self.flag = np.zeros(self.Nint, dtype=int)

        for t_idx in range(Nt_total):
            # Real time
            t_real = t_idx * self.dt

            # Update flags based on current time
            for i in range(self.Nint):
                if (t_real > self.interval_start[i]) and (t_real <= self.interval_start[i] + self.T/2):
                    self.flag[i] = 1  # first half of interval
                elif (t_real > self.interval_start[i] + self.T/2) and (t_real <= self.interval_start[i] + self.T):
                    self.flag[i] = 2  # second half of interval
                else:
                    self.flag[i] = 0  # inactive

            # Assign velocity at current timestep
            self.ur = vel_ds['UVEL'].isel(time=t_idx).values
            self.vr = vel_ds['VVEL'].isel(time=t_idx).values

            # Advance ξ maps according to strategy and current flags
            for interval in range(self.Nint):
                if self.flag[interval] == 0:
                    continue  # skip inactive intervals

                if self.strat == 1:
                    if self.solve_xi12:
                        self._timestep_RK4_xi12(interval)
                    if self.solve_xi13:
                        self._timestep_RK4_xi13(interval)
                elif self.strat == 2:
                    self._timestep_RK4_xi21(interval)
                    if self.solve_xi23:
                        self._timestep_RK4_xi23(interval)
                elif self.strat == 3:
                    self._timestep_xi31(interval)
                    if self.solve_xi32:
                        self._timestep_RK4_xi32(interval)

            # Update Lagrangian mean fields from current velocities
            self._update_LM_fields_from_velocity(self.ur, self.vr)

            # Save timestep for LM
            self.timesteps_LM.append(t_idx)

            # Update kernel arrays if not exponential
            if self.kernel_type != 'exponential':
                for interval in range(self.Nint):
                    self.kernel_current[interval], self.kernel_current_int[interval] = self._kernel(
                        t_real - self.interval_start[interval], kernel_type=self.kernel_type)

        # Finalise Lagrangian mean fields in real space
        self.z_LMr = irfft2(self.z_LMk)

        # Regrid LM fields according to strategy
        if self.strat == 1:
            self.z_LMr_at_end = self.z_LMr
            if self.solve_xi12:
                Xi12xr = irfft2(self.xi12xk) + self.xx
                Xi12yr = irfft2(self.xi12yk) + self.yy
                self.z_LMr_at_mean = self._regrid_scalar_nd(self.z_LMr, Xi12xr, Xi12yr)
            if self.solve_xi13:
                Xi13xr = irfft2(self.xi13xk) + self.xx
                Xi13yr = irfft2(self.xi13yk) + self.yy
                self.z_LMr_at_mid = self._regrid_scalar_nd(self.z_LMr, Xi13xr, Xi13yr)
        elif self.strat == 2:
            Xi21xr = irfft2(self.xi21xk) + self.xx
            Xi21yr = irfft2(self.xi21yk) + self.yy
            self.z_LMr_at_mean = self.z_LMr
            self.z_LMr_at_end = self._regrid_scalar_nd(self.z_LMr, Xi21xr, Xi21yr)
            if self.solve_xi23:
                Xi23xr = irfft2(self.xi23xk) + self.xx
                Xi23yr = irfft2(self.xi23yk) + self.yy
                self.z_LMr_at_mid = self._regrid_scalar_nd(self.z_LMr, Xi23xr, Xi23yr)
        elif self.strat == 3:
            Xi31xr = irfft2(self.xi31xk) + self.xx
            Xi31yr = irfft2(self.xi31yk) + self.yy
            self.z_LMr_at_mid = self.z_LMr
            self.z_LMr_at_end = self._regrid_scalar_nd(self.z_LMr, Xi31xr, Xi31yr)
            if self.solve_xi32:
                Xi32xr = irfft2(self.xi32xk) + self.xx
                Xi32yr = irfft2(self.xi32yk) + self.yy
                self.z_LMr_at_mean = self._regrid_scalar_nd(self.z_LMr, Xi32xr, Xi32yr)

        t_end = time.time()
        print(f'Offline GLM solve completed in {t_end - t_start:.2f} s')


