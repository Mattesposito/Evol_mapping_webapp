import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import camb
from camb import model
import numpy as np
from scipy.interpolate import interp1d

# plt.rcParams.update({"text.usetex": True, "font.family": "serif"})   # Doesn't work with streamlit
##########      Set the page configuration     ###########
st.set_page_config(
    page_title="Evolution mapping",
    page_icon=":arrows_counterclockwise:",
)
secondaryBackgroundColor="#F0F2F6"
##########################################################

## Define the function to get the linear power spectrum from camb
@st.cache_data(show_spinner=False)
def get_Pk_camb(param,z=None,npoints=1000, kmin=None, kmax=None, Mpc_units=True, nz=100):
    ### set up the redshifts and parameters
    pars = camb.CAMBparams()
    redshifts=np.linspace(0.5, 1.5, nz+1) if z is None else [z]
    pars.set_cosmology(H0=param['h0']*100.,
                    ombh2=param['ombh2'],
                    omch2=param['omch2'],
                    omk=param['Omk'],
                    num_massive_neutrinos=1,
                    mnu=93.14 * param['omnuh2'],
                    standard_neutrino_neff=2.0328)
    pars.InitPower.set_params(ns=param['ns'],
                                As=param['As']) 
    if param['w'] != -1. or param['wa'] != 0.:
        pars.set_dark_energy(w=param['w'], wa=param['wa'])
    #Linear spectra  
    pars.NonLinear = model.NonLinear_none
    pars.DoLensing = False
    kmin = 3e-3 if kmin is None else kmin
    kmax = 50 if kmax is None else kmax
    pars.set_matter_power(redshifts=redshifts, kmax=kmax)   #This kmax is always in Mpc^-1
    results = camb.get_results(pars)
    if Mpc_units:
        kmin /= param['h0']
        kmax /= param['h0']

    kh, redshifts, pk_lin = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = npoints)
    pk_lin = pk_lin[nz//2] if z is None else pk_lin[0]
    print(redshifts)

    sigma_8 = results.get_sigmaR(8, hubble_units=True)
    sigma_12 = results.get_sigmaR(12, hubble_units=False)
    print(sigma_12)
    if z is not None:
        sigma_8 = sigma_8[0]
        sigma_12 = sigma_12[0]
    
    if Mpc_units:
        kh *= param['h0']
        pk_lin /= param['h0']**3
    return kh, pk_lin, sigma_8, sigma_12, redshifts


##########       Choosing the parameters and units     ###########
param_space = st.sidebar.selectbox(
    'Do you want good parameters or bad parameters?',
    ('Good parameters', 'Bad parameters')
)
units = st.sidebar.selectbox(
    'Do you want good units or bad units?',
    ('Good units (Mpc)', 'Bad units (Mpc/h)')
)

Mpc_units = True if units == 'Good units (Mpc)' else False
phys_param = True if param_space == 'Good parameters' else False
# Mpc_units = False
# phys_param = True
##################################################################


############       Define the default parameters     #############
param_def = {'h0': 0.67, 'Omega_m': 0.45570, 'ombh2': 0.02235, 
                 'omch2': 0.191692, 'omnuh2': 0.0006, 
                 'As': 1.70e-09, 'ns': 0.96,
                 'w': -1., 'wa': 0., 'Omk': 0}
z_def = 1.0
k_def, Pk_def, sigma_8_def, sigma_12_def, _ = get_Pk_camb(param_def, z=z_def, Mpc_units=Mpc_units)
h0_def = param_def['h0']
##################################################################


########       Set the sliders for the parameters     ############
help_fix_param = 'Either the redshift will be adjusted \n\n ' + \
    r' to reach a given $\sigma_{12}$ or $\sigma_8$ value, ' + '\n\n' + \
    r'or the $\sigma_{12}$ and $\sigma_8$ value will be ' + \
        ' \n\n calculated at a given redshift.'
help_fix_param = None

param_names_latex = {'h0': r'$h$', 'ombh2': r'$\omega_b$', 'omch2': r'$\omega_{\mathrm{CDM}}$',
                    'omnuh2': r'$\omega_{\nu}$', 'As': r'$A_s$', 'ns': r'$n_s$', 
                    'w': r'$w$', 'wa': r'$w_a$', 'omkh2': r'$\Omega_k$'}

with st.sidebar:
    # st.write(r'## Show $P_\mathrm{L}(k)$ at a given...')
    fix_param = st.segmented_control(r'Show $P_\mathrm{L}(k)$ at a given...', [r'$\sigma_{12}$', r'$\sigma_8$', r'$z$'], 
                                     selection_mode="single", default=r'$\sigma_{12}$', label_visibility='visible',
                                     help=help_fix_param)
    var_param = st.segmented_control(r'Select parameter to vary', 
                                     ['h0', 'omch2', 'As', 'w', 'omkh2'],
                                    selection_mode="single", default='As', label_visibility='visible',
                                    format_func=lambda x: param_names_latex[x])
    if var_param == 'As':
        As = st.slider(
                    r'Select a value for $A_s [10^{-9}]$',
                    0.8*1.70, 1.2*1.70, 1.70, step=0.01, format='%.2f'
                )
        As *= 1e-9
    elif var_param == 'w':
        w = st.slider(
                    r'Select a value for $w$',
                    -1.2, -0.8, -1., step=0.01, format='%.2f'
                )
    elif var_param == 'omkh2':
        omkh2 = st.slider(
                    r'Select a value for $\omega_K$',
                    -0.05*0.67, 0.05*0.67, 0., step=0.01, format='%.2f'
                )
        Omk = omkh2/h0_def**2
        var_param = 'Omk'
    elif var_param == 'h0':
        h0 = st.slider(
                    r'Select a value for $h$',
                    0.3, 1.2, 0.67, step=0.01, format='%.2f'
                )
    elif var_param == 'omch2':
        omch2 = st.slider(
                r'Select a value for $\omega_{\mathrm{{CDM}}}$',
                0.8*(0.191692), 1.2*(0.191692), (0.191692), step=0.005, format='%.5f'
            )
    else:
        pass
##################################################################

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

##############       Calculate the camb Pk     ###################
with _lock:
    # param = {'h0': h0, 'ombh2': ombh2, 'omch2': omch2, 
    #              'omnuh2': omnuh2, 'As': As, 'ns': ns,
    #              'w': w, 'wa': wa, 'Omk': Omk}
    # param = {'h0': 0.67, 'Omega_m': 0.45570, 'ombh2': 0.02235, 
    #                 'omch2': 0.191692, 'omnuh2': 0.0006, 
    #                 'As': As, 'ns': 0.96,
    #                 'w': -1., 'wa': 0., 'Omk': 0}
    param = {**param_def}
    param[var_param] = locals()[var_param]
    nz = 100

    k, Pk, sigma_8_z, sigma_12_z, z = get_Pk_camb(param, Mpc_units=Mpc_units, nz=nz)#get_Pk_camb(param, z=z, Mpc_units=Mpc_units)
    sigma_8 = sigma_8_z[nz//2]
    sigma_12 = sigma_12_z[nz//2]
    z = z[::-1]      # The redshifts are re-ordered before calculating the sigma_R
    print('pre', sigma_12, sigma_8, 'tar:', sigma_12_def, sigma_8_def)
    if fix_param == r'$\sigma_{12}$':
        z_tar = np.interp(sigma_12_def, sigma_12_z, z)#interp1d(sigma_12_z, z)(sigma_12_def)#
        k, Pk, sigma_12, sigma_8, _ = get_Pk_camb(param, z=z_tar, Mpc_units=Mpc_units)
    elif fix_param == r'$\sigma_8$':
        z_tar = np.interp(sigma_8_def, sigma_8_z, z)
        k, Pk, sigma_12, sigma_8, _ = get_Pk_camb(param, z=z_tar, Mpc_units=Mpc_units)
    else:
        z_tar = z_def
    print('post', sigma_12, sigma_8, 'tar:', sigma_12_def, sigma_8_def)
    print('z_tar', z_tar, 'z_def', z_def)
##################################################################



with _lock:
    with st.sidebar:
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor(secondaryBackgroundColor)
        plt.plot(sigma_12_z, z, ls='-', c='k')
        plt.ylabel(r'$z$', fontsize=18)
        plt.xlabel(r'$\sigma_{12}$', fontsize=18)
        plt.tight_layout()
        plt.axvline(sigma_12_def, ls='--', c='red')
        plt.axhline(z_def, ls='--', c='red')
        plt.axvline(sigma_12, ls='--', c='k')
        plt.axhline(z_tar, ls='--', c='k')
        plt.xticks_params = {'fontsize': 18, 'direction': 'in'}
        plt.yticks_params = {'fontsize': 18, 'direction': 'in'}
        plt.xlim(0.44, 0.64)
        plt.ylim(0.5, 1.5)
        st.pyplot(fig)
    # print(sigma_12)

with _lock:
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)

    # Plot on the first (main) axes
    ax0.loglog(k_def, Pk_def, ls='--', c='k')
    ax0.loglog(k, Pk, c='#FF4B4B')
    if Mpc_units:
        ax1.set_xlabel(r'k/($\mathrm{Mpc}^{-1}$)', fontsize=14)
        ax0.set_ylabel(r'P(k)/($\mathrm{Mpc}^{3}$)', fontsize=14)
        ax0.set_xlim(3e-3, 1)
        ax0.set_ylim(1e2, 8e4)
    else:
        ax1.set_xlabel(r'k/(h$\mathrm{Mpc}^{-1}$)', fontsize=14)
        ax0.set_ylabel(r'P(k)/($\mathrm{Mpc}/h)^{3}$', fontsize=14)
        ax0.set_xlim(3e-3, 1)
        ax0.set_ylim(1e2, 2.5e4)

    ax0.text(0.05, 0.91, r'fix $\sigma_{12}$ = '+ f'{sigma_12:.4f}', c='red' if fix_param == r'$\sigma_{12}$' else 'k',
             fontsize=12, transform=ax0.transAxes)
    ax0.text(0.81, 0.91, r'$z$ = '+ f'{z_tar:.4f}', c='red' if fix_param == r'$z$' else 'k',
             fontsize=12, transform=ax0.transAxes)

    # Calculate residuals (example, adjust according to your data)
    residuals = (Pk/Pk_def-1)*100
    residuals[np.abs(residuals)<1e-10] = 0
    if np.max(np.abs(residuals)) < 0.1:
        ax1.set_ylim(-0.1, 0.1)

    # Plot on the second (residuals) axes
    ax1.axhline(0, ls='--', c='k')
    ax1.plot(k, residuals, ls='-', c='#FF4B4B')
    ax1.set_ylabel('Residuals (%)', fontsize=14)
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)  # Adjust space between the plots
    plt.setp(ax0.get_xticklabels(), visible=False)  # Hide ax0's x-tick labels

    # Saving figure in case the user wants to download it
    plot_name = 'Pk_plot.pdf'
    plt.savefig(plot_name)

    st.pyplot(fig)


st.write('\n')
st.write('\n')
with open(plot_name, "rb") as img:
    btn = st.download_button(
        label="Download plot",
        data=img,
        file_name=plot_name,
        mime="image/pdf"
    )
st.write('\n')
st.write('\n')
st.markdown('Powered with **camb**, by Antony Lewis and Anthony Challinor (https://github.com/cmbant/CAMB)')