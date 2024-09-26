import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import camb
from camb import model
import numpy as np
import os

st.set_page_config(
    page_title="Create linear Pk GIF",
    page_icon=":floppy_disk:",
)

# def set_plot_params(labels = True):
#     plt.rcParams.update({'axes.labelsize': 'x-large'})
#     plt.rcParams['figure.figsize'] = [12, 12]
#     plt.rcParams['figure.dpi'] = 200
#     plt.rcParams['axes.linewidth'] = 2.3
#     plt.tick_params(reset = True, which='both', bottom=True, top=True, right=True, left=True, 
#                     direction = 'in', length=8.5, width=2.3, labelsize = 18,
#                     labelleft=labels, labeltop=False, labelright=False, labelbottom=labels)
#     plt.tick_params(which='major', bottom=True, top=True, right=True, left=True, 
#                     direction = 'in', length=17, width=2.3)


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

def get_Pk_var(var, c):
    param_def = {'h0': 0.67, 'Omega_m': 0.45570, 'ombh2': 0.02235, 
                 'omch2': 0.191692, 'omnuh2': 0.0006, 
                 'As': 1.70e-09, 'ns': 0.96,
                 'w': -1., 'wa': 0., 'Omk': 0}
    
    if var == 'Omk':
        eps = c
    else:
        eps = param_def[var]*c#cosmo_dict[base_cosmo][var]*c
    param = {**param_def, var: param_def[var]+eps}
    if var == 'ombh2':
        param['omch2'] = param['omch2']-eps
    return get_Pk_camb(param)


def save_animation(var_param, var_range, smooth):
    # We first delete the older gifs
    os.system('rm -f *.gif')
    Pk_list = [get_Pk_var(var_param, c) for c in np.linspace(0, -var_range, smooth)]
    Pk_list += Pk_list[::-1][:-1]
    temp = [get_Pk_var(var_param, c) for c in np.linspace(0, var_range, smooth)]
    Pk_list += temp + temp[::-1]

    def init_anim():
        # set_plot_params()
        plt.xlim(0.006314601343965478, 1)
        plt.ylim(3.15e1*2, 2e3*2)
        p = plt.loglog(Pk_list[0][0], Pk_list[0][0]*Pk_list[0][1], c='black', ls='--', dashes=(6.5, 5.5), linewidth=2.2)# **A_plot_kwargs('--'))

    def run_anim(Pk):
    #     plt.clf()
        p[0].set_data(Pk[0], Pk[0]*Pk[1])
        # set_plot_params()
        # set_Ariel_like_plots(set_lim=False)
        plt.xlim(0.006314601343965478, 1)
        plt.ylim(3.15e1*2, 2e3*2)
        return p

    # fig = plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots(figsize=(12, 12))
    _ = [spine.set_linewidth(2.3) for spine in ax.spines.values()]
    ax.tick_params(reset = True, which='both', bottom=True, top=True, right=True, left=True, 
                    direction = 'in', length=8.5, width=2.3, labelsize = 24,
                    labelleft=True, labeltop=False, labelright=False, labelbottom=True, pad=8)
    ax.tick_params(which='major', bottom=True, top=True, right=True, left=True, 
                    direction = 'in', length=17, width=2.3)
    # set_plot_params()
    # p = plt.loglog([], [], c='black', linewidth=2.2)#, **A_plot_kwargs())
    p = ax.loglog([], [], c='black', linewidth=2.2)
    ax.set_xlabel(r'k/($\mathrm{Mpc}^{-1}$)', fontsize = 24)
    ax.set_ylabel(r'k $\times$ P(k)/($\mathrm{Mpc}^{2}$)', fontsize = 24)
    ax.text(0.65, 0.04, 'Credits: Matteo Esposito', c='grey', fontsize=16, transform=ax.transAxes)
    fig.patch.set_alpha(0.)
    fig.tight_layout(pad = 2)
    ani = animation.FuncAnimation(fig, run_anim, Pk_list[::2]+[Pk_list[0]], interval=50, init_func=init_anim)
    # components.html(ani.to_jshtml(), height=10)
    ani.save(f'var_{var_param}.gif', savefig_kwargs={"transparent": True})#, writer=animation.PillowWriter())#, 'facecolor': None})#, writer=animation.PillowWriter())
    st.image(f'var_{var_param}.gif')
    # return f'var_{var_param}.gif'


st.title('Choose the parameter you want to vary and download your GIF')
st.title('\n')

# Add a selectbox to the sidebar:
var_param = st.sidebar.selectbox(
    'Which parameter do you want to vary?',
    ('ombh2', 'omch2',
     'ns', 'As', 'w', 'wa',
     'Omk', 'h0')
)

param_label = {'ombh2': r'$\omega_b$', 'omch2': r'$\omega_{\mathrm{{CDM}}}$',
     'ns': r'$n_s$', 'As': r'$A_s$', 'w': r'$w$', 'wa': r'$w_a$',
     'Omk': r'$\Omega_K$', 'h0': r'$h', 'omnuh2': r'$\omega_{\nu}$'}[var_param]


omnuh2_list = [0, 0.0001, 0.0006, 0.001, 0.01, 0.1]
omkh2 = 0.
h0 = 0.67
with st.sidebar:
    if var_param == 'Omk':
        var_range = st.slider(
            r'Choose the increase/decrease range in which to vary %s (in absolute value)' %param_label,
            0.05, 0.3, 0.1, step=0.05, format='%.2f'
        )
    else:
        var_range = st.slider(
            r'Choose the increase/decrease range in which to vary %s (in relative value)' %param_label,
            0.1, 0.6, 0.4, step=0.1, format='%.1f'
        )
    smoothness = st.slider(
            r'Choose the smoothness for the animation (in number of frames)',
            40, 200, 120, step=20
        )
    nframes_per_loop = smoothness//4

# create_btn = False
# create_btn = st.button(label='Create GIF', on_click=save_animation, args=(var_param, 0.4))
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

with _lock:
    save_animation(var_param, var_range, nframes_per_loop)
gif_fname = f'var_{var_param}.gif'

with open(gif_fname, "rb") as img:
    dwnl_btn = st.download_button(
        label="Download GIF",
        data=img,
        file_name=gif_fname,
        mime="image/gif"
    )

    st.write('\n')
st.write('\n')
st.markdown('Powered with **camb**, by Antony Lewis and Anthony Challinor (https://github.com/cmbant/CAMB)')