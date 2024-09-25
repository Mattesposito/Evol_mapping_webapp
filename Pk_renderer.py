import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import camb
from camb import model
import numpy as np

@st.cache_data(show_spinner=False)
def get_Pk_camb(param,z=0,npoints=1000, kmin=None, kmax=None, Mpc_units=True):
    ### set up the redshifts and parameters
    pars = camb.CAMBparams()
    redshifts=[z]
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
    pars.set_matter_power(redshifts=redshifts)
    #Linear spectra  
    pars.NonLinear = model.NonLinear_none
    pars.DoLensing = False
    results = camb.get_results(pars)
    kmin = 1e-4 if kmin is None else kmin
    kmax = 1 if kmax is None else kmax
    if Mpc_units:
        kmin /= param['h0']
        kmax /= param['h0']

    kh, z, pk_lin = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = npoints)
    pk_lin = pk_lin[0]
    if Mpc_units:
        kh *= param['h0']
        pk_lin /= param['h0']**3
    return kh,pk_lin


st.title('Fantastic parameters and where to find them')
st.title('\n')

# Add a selectbox to the sidebar:
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

if phys_param:
    omnuh2_list = [0, 0.0001, 0.0006, 0.001, 0.01, 0.1]
    with st.sidebar:
        with st.expander("Shape parameters", expanded=True):
            ombh2 = st.slider(
                r'Select a value for $\omega_b$',
                0.6*0.02235, 1.4*0.02235, 0.02235, step=0.001, format='%.5f'
            )

            omch2 = st.slider(
                r'Select a value for $\omega_{\mathrm{{CDM}}}$',
                0.8*(0.191692-(ombh2-0.02235)), 1.2*(0.191692-(ombh2-0.02235)), (0.191692-(ombh2-0.02235)), step=0.005, format='%.5f'
            )

            ns = st.slider(
                r'Select a value for $n_s$',
                0.8, 1.2, 0.96, step=0.01, format='%.2f'
            )
        with st.expander("Evolution parameters", expanded=True):
            As = st.slider(
                r'Select a value for $A_s [10^{-9}]$',
                0.8*1.70, 1.2*1.70, 1.70, step=0.01, format='%.2f'
            )
            As *= 1e-9

            wa = st.slider(
                r'Select a value for $w_a$',
                -0.20, 0.20, 0., step=0.01, format='%.2f'
            )
            min_w = max(-1., -1.-wa) + 1e-5 if wa != 0. else -1.5

            w = st.slider(
                r'Select a value for $w$',
                min_w, -0.5, max(-1., min_w), step=0.01, format='%.2f'
            )

            omkh2 = st.slider(
                r'Select a value for $\omega_K$',
                -0.05*0.67, 0.05*0.67, 0., step=0.01, format='%.2f'
            )

            h0 = st.slider(
                r'Select a value for $h$',
                0.5, 1.0, 0.67, step=0.01, format='%.2f'
            )
            
            Omk = omkh2/h0**2

        with st.expander("Mixed parameters", expanded=False):
            omnuh2 = st.select_slider(
                label=r'Select a value for $\omega_\nu$',
                options=omnuh2_list, value=0.0006
            )
else:
    Omnu_list = [0, 0.0001/0.67**2, 0.0006/0.67**2, 0.001/0.67**2, 0.01/0.67**2, 0.1/0.67**2]
    
    Omb = st.sidebar.slider(
        r'Select a value for $\Omega_b$',
        0.8*0.02235/0.67**2, 1.2*0.02235/0.67**2, 0.02235/0.67**2, step=0.001, format='%.5f'
    )

    Omc = st.sidebar.slider(
        r'Select a value for $\Omega_{\mathrm{{CDM}}}$',
        0.8*(0.191692-(Omb*0.67**2-0.02235))/0.67**2, 1.2*(0.191692-(Omb*0.67**2-0.02235))/0.67**2, (0.191692-(Omb*0.67**2-0.02235))/0.67**2, step=0.005, format='%.5f'
    )

    Omnu = st.sidebar.select_slider(
        label=r'Select a value for $\Omega_\nu$',
        options=Omnu_list, value=0.0006/0.67**2, format_func=lambda x: '%.5f' %x,
    )

    As = st.sidebar.slider(
        r'Select a value for $A_s [10^{-9}]$',
        0.8*1.70, 1.2*1.70, 1.70, step=0.01, format='%.2f'
    )
    As *= 1e-9

    ns = st.sidebar.slider(
        r'Select a value for $n_s$',
        0.8, 1.2, 0.96, step=0.01, format='%.2f'
    )

    h0 = st.sidebar.slider(
        r'Select a value for $h$',
        0.5, 1.0, 0.67, step=0.01, format='%.2f'
    )

    wa = st.sidebar.slider(
        r'Select a value for $w_a$',
        -0.20, 0.20, 0., step=0.01, format='%.2f'
    )
    min_w = max(-1., -1.-wa) + 1e-5 if wa != 0. else -1.5

    w = st.sidebar.slider(
        r'Select a value for $w$',
        min_w, -0.5, max(-1., min_w), step=0.01, format='%.2f'
    )

    Omk = st.sidebar.slider(
        r'Select a value for $\Omega_K$',
        -0.05, 0.05, 0., step=0.01, format='%.2f'
    )

    ombh2 = Omb*h0**2
    omch2 = Omc*h0**2
    omnuh2 = Omnu*h0**2

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

with _lock:
    param = {'h0': h0, 'ombh2': ombh2, 'omch2': omch2, 
                 'omnuh2': omnuh2, 'As': As, 'ns': ns,
                 'w': w, 'wa': wa, 'Omk': Omk}
    k, Pk = get_Pk_camb(param, Mpc_units=Mpc_units)

    param_def = {'h0': 0.67, 'Omega_m': 0.45570, 'ombh2': 0.02235, 
                 'omch2': 0.191692, 'omnuh2': 0.0006, 
                 'As': 1.70e-09, 'ns': 0.96,
                 'w': -1., 'wa': 0., 'Omk': 0}
    k_def, Pk_def = get_Pk_camb(param_def, Mpc_units=Mpc_units)

    print(param)

    # fig, ax = plt.subplots()
    # plt.tick_params(reset = True, which='both', bottom=True, top=True, right=True, left=True, 
    #                 direction = 'in',
    #                 labelleft=True, labeltop=False, labelright=False, labelbottom=True)
    # ax.loglog(k_def, Pk_def, ls = '--', c='coral')
    # ax.loglog(k, Pk)
    # if Mpc_units:
    #     ax.set_xlabel(r'k/($\mathrm{Mpc}^{-1}$)')
    #     ax.set_ylabel(r'P(k)/($\mathrm{Mpc}^{3}$)')
    #     ax.set_xlim(6e-4, 1.5)
    #     ax.set_ylim(1e2, 8e4)
    # else:
    #     ax.set_xlabel(r'k/(h$\mathrm{Mpc}^{-1}$)')
    #     ax.set_ylabel(r'P(k)/($\mathrm{Mpc}/h)^{3}$')
    #     ax.set_xlim(6e-4, 1.5)
    #     ax.set_ylim(1e2, 2.5e4)

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)

    # Plot on the first (main) axes
    ax0.loglog(k_def, Pk_def, ls='--', c='coral')
    ax0.loglog(k, Pk)
    if Mpc_units:
        ax1.set_xlabel(r'k/($\mathrm{Mpc}^{-1}$)')
        ax0.set_ylabel(r'P(k)/($\mathrm{Mpc}^{3}$)')
        ax0.set_xlim(1e-4, 1)
        ax0.set_ylim(1e2, 8e4)
    else:
        ax1.set_xlabel(r'k/(h$\mathrm{Mpc}^{-1}$)')
        ax0.set_ylabel(r'P(k)/($\mathrm{Mpc}/h)^{3}$')
        ax0.set_xlim(1e-4, 1)
        ax0.set_ylim(1e2, 2.5e4)

    # Calculate residuals (example, adjust according to your data)
    residuals = (Pk/Pk_def-1)*100
    residuals[residuals<1e-10] = 0

    # Plot on the second (residuals) axes
    ax1.plot(k, residuals, ls='-', c='darkviolet')
    ax1.set_ylabel('Residuals (%)')
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