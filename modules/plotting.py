import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from modules.data_loading import *
from modules.data_processing import *
from parameters.data_parameters import *
from parameters.experimental_parameters import *
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
import glob
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ipywidgets import interact


#Plot triggers (enable/disable for extra plots):

# Delete all stored images in Plots subfolder before saving new
delete_png_before_use = True

# Simple pT and rapidity plots
preliminary_plots = False

# Dynamic plot (Web hosted 3d lab frame decay vertex plot with daughters as arrows)
dynamic_plot = False

# Decay vertex statistics for specific mass/mix HNLs (configure below by plot)
plots_decay_stats = False



# Functions:

def delete_png_files(folder_path):
    """
    Deletes all .png files in the specified folder.

    Parameters:
    - folder_path: A string representing the path to the folder from which .png files will be deleted.
    """
    # Create a pattern to match all .png files in the folder
    pattern = os.path.join(folder_path, '*.png')

    # Find all files in the folder that match the pattern
    png_files = glob.glob(pattern)

    # Iterate over the list of file paths and remove each file
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def enable_close_with_esc_key(fig):
    """
    Allows closing the figure window by pressing the Escape key.

    :param fig: A matplotlib figure object to attach the close event.
    """
    def close_event(event):
        if event.key == 'escape':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', close_event)


def momentum_distribution(batch, batch_mass='1 GeV'):
    bin_number = 100
    plt.title(f'Transverse momentum distribution for {batch_mass} HNL events')
    plt.ylabel('Frequency')
    plt.xlabel('Transverse momentum (GeV)')
    plt.hist(abs(batch.mass(batch_mass).particle('boson').pT()),           histtype='step', bins=bin_number, label ='boson', linestyle='--')
    plt.hist(abs(batch.mass(batch_mass).particle('hnl').pT()),             histtype='step', bins=bin_number, label ='hnl', linestyle='-')
    plt.hist(abs(batch.mass(batch_mass).particle('prompt').pT()),          histtype='step', bins=bin_number, label ='tau', linestyle='--')
    plt.hist(abs(batch.mass(batch_mass).particle('displaced_minus').pT()), histtype='step', bins=bin_number, label ='mu-', linestyle='-')
    plt.hist(abs(batch.mass(batch_mass).particle('displaced_plus').pT()),  histtype='step', bins=bin_number, label ='mu+', linestyle='-')
    plt.hist(abs(batch.mass(batch_mass).particle('neutrino').pT()),        histtype='step', bins=bin_number, label ='vt', linestyle='dotted')
    plt.yscale('log')
    plt.legend()
    plt.show()


def plot_histograms(data_list, title, x_label, y_label, savename='', bin_number=100, logplot=True):
    """
    Plots a series of histograms with dynamic inputs ensuring the same bin widths for all histograms.

    :param data_list: A list of dictionaries, each containing:
                      - 'data': The array of data points to plot.
                      - 'label': Label for the histogram.
                      - 'linestyle': The linestyle for the histogram.
    :param title: The title of the plot.
    :param x_label: The label for the x-axis.
    :param y_label: The label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if logplot:
        plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.25)

    # Find global min and max
    global_min = min(entry['data'].min() for entry in data_list)
    global_max = max(entry['data'].max() for entry in data_list)

    # Create uniform bin edges
    bins = np.linspace(global_min, global_max, bin_number+1)

    for entry in data_list:
        plt.hist(entry['data'], bins=bins, histtype='step', label=entry['label'], linestyle=entry.get('linestyle', '-'))

    plt.legend()
    if savename != '':
        save_plot(savename)
    plt.show()



def plot_survival_3d(survival_array, cut_type):
    plt.figure(figsize=(12, 8))

    survival_array_percent = survival_array * 100
    mass_edges = np.linspace(mass_hnl[0] - (mass_hnl[1] - mass_hnl[0])/2, 
                            mass_hnl[-1] + (mass_hnl[1] - mass_hnl[0])/2, 
                            len(mass_hnl) + 1)
    mixing_edges_log = np.linspace(np.log10(mixing[0]) - (np.log10(mixing[1]) - np.log10(mixing[0]))/2, 
                                np.log10(mixing[-1]) + (np.log10(mixing[1]) - np.log10(mixing[0]))/2, 
                                len(mixing) + 1)

    # Plot the heatmap
    plt.pcolormesh(mass_edges, mixing_edges_log, survival_array_percent.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Survival Percentage (%)')

    # Adjusting the plot
    plt.title(f'Survival Percentage of $\\mu^\\pm$ after {cut_type}')
    plt.xlabel('HNL Mass $M_N$ (GeV)')
    plt.ylabel('Mixing Angle $\\Theta_{\\tau}^2$')

    # Define y-ticks and labels to represent standard spacing in log scale
    major_ticks = np.arange(0, -9, -1)  # From 0 to -8
    minor_ticks = np.array([-0.3, -0.7])  # For minor ticks at 1e-0.3 and 1e-0.7, adjust as needed

    # Set major and minor ticks
    plt.yticks(major_ticks, labels=[f"1e{val}" for val in major_ticks])

    # Optionally, add minor ticks for better granularity
    from matplotlib.ticker import AutoMinorLocator
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(10))  # Adjust 10 to the number of minor intervals you want
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

    plt.show()

def plot_survival_2d(survival_array, label_array):
    """
    Plot survival percentages as a function of HNL mass.

    Parameters:
    - mass_hnl: Array of HNL masses.
    - survival_percentages: Array of survival percentages corresponding to each HNL mass.
    """
    plt.figure(figsize=(10, 6))
    for i in range(len(survival_array)):
        plt.plot(mass_hnl, survival_array[i], label=f'{label_array[i]}')
    
    # Adding plot decorations
    plt.title(f'Survival Percentages of $\\mu^\\pm$')
    plt.xlabel('HNL Mass $M_N$ (GeV)')
    plt.ylabel('Survival Percentage (%)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.show()

def plot_parameter_space_region(production_allcuts):
    # Ensure production_minimum is set to a meaningful threshold based on your context
    mass_grid, mixing_grid = np.meshgrid(np.linspace(min(mass_hnl), max(mass_hnl), 100),
                                         np.logspace(np.log10(min(mixing)), np.log10(max(mixing)), 100))

    points = np.array([np.repeat(mass_hnl, len(mixing)), np.tile(mixing, len(mass_hnl))]).T
    values = production_allcuts.flatten()

    # Interpolate onto the grid
    production_grid = griddata(points, values, (mass_grid, mixing_grid), method='linear')

    plt.figure(figsize=(10, 6))
    
    levels = np.linspace(0, values.max(), 5) # for continuous use np.linspace(0, values.max(), 100) and for discrete use np.array([0, production_minimum, values.max()])
    contour_filled = plt.contourf(mass_grid, mixing_grid, production_grid, levels=levels, extend='max', cmap='Greens')
    plt.colorbar(contour_filled, label='Production')

    # Highlight the region where production meets or exceeds the minimum threshold
    # Assuming production_minimum is the value above which the region is highlighted
    contour = plt.contour(mass_grid, mixing_grid, production_grid, levels=[production_minimum], colors='red', linewidths=2, linestyles='-')

    plt.text(mass_grid.max()-5, mixing_grid.max()-5, f'Production > {production_minimum}', color='red', fontsize=10, backgroundcolor='white')
    plt.clabel(contour, inline=True, fontsize=8)

    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('HNL Mass, $M_N$ (GeV)', size=12)
    plt.ylabel('Mixing, $\\Theta_{\\tau}^2$', size=12)
    plt.grid(alpha=0.25)
    plt.title(f'HNL production parameter space')
    save_plot('hnl_production_parameter_space_contour')
    plt.show()

def plot_parameter_space_regions(*production_arrays, labels=None, colors=None, smooth=False, sigma=1):
    """
    Plot multiple production arrays on the same parameter space with contours.

    :param production_arrays: Unpacked tuple of production arrays.
    :param labels: Optional list of labels for each production array's contour.
    :param colors: Optional list of colors for each contour line.
    :param smooth: Apply Gaussian smoothing to the last production array.
    :param sigma: Standard deviation for Gaussian kernel if smoothing is applied.
    """
    mass_grid, mixing_grid = np.meshgrid(
        np.linspace(min(mass_hnl), max(mass_hnl), 100),
        np.logspace(np.log10(min(mixing)), np.log10(max(mixing)), 100)
    )

    plt.figure(figsize=(10, 6))

    if labels is None:
        labels = [f'Contour {i+1}' for i in range(len(production_arrays))]
    if colors is None:
        colors = plt.cm.jet(np.linspace(0, 1, len(production_arrays)))

    for i, production_array in enumerate(production_arrays):
        if smooth and i == len(production_arrays) - 1:  # Apply smoothing to the last array if smooth is True
            values = gaussian_filter(production_array, sigma=sigma).flatten()
        else:
            values = production_array.flatten()

        production_grid = griddata(
            (np.repeat(mass_hnl, len(mixing)), np.tile(mixing, len(mass_hnl))),
            values,
            (mass_grid, mixing_grid),
            method='linear'
        )

        contour = plt.contour(
            mass_grid, mixing_grid, production_grid,
            levels=[production_minimum],
            colors=[colors[i]],
            linewidths=2,
            linestyles='-'
        )

        # Plot invisible line for legend entry
        plt.plot([], [], color=colors[i], linewidth=2, linestyle='-', label=labels[i])

    plt.legend(loc='upper right', frameon=True)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('HNL Mass, $M_N$ (GeV)', size = 12)
    plt.ylabel('Mixing, $\\Theta_{\\tau}^2$', size = 12)
    plt.title('HNL Production Parameter Space')
    plt.grid(alpha=0.25)
    save_plot('hnl_production_parameter_space_multi')
    plt.show()

def mean_from_2d_survival(arr):
    temp_array = [np.mean(i) for i in arr]
    return temp_array

def save_plot(name, dpi=200):
    current_directory = os.getcwd()                                         # Current path
    data_path         = os.path.join(current_directory,'data', data_folder) # Data folder path
    plot_path         = os.path.join(data_path, 'Plots', f'{name}.png')
    os.makedirs(os.path.join(data_path, 'Plots'), exist_ok=True) # Making directory if not already exists

    plt.savefig(plot_path, dpi=dpi)
    print(f"Plot saved to {plot_path}")



def plot_decay_vertices_with_trajectories(r_labs, survival_bool, momenta, index_mass, index_mixing, title='Decay Vertices and Trajectories'):
    """
    Plot the decay vertices in 3D space with trajectories of daughter particles for given mass and mixing indices.

    :param r_labs: An array of decay vertices with shape (masses, mixings, particles, 3).
    :param survival_bool: A boolean array indicating survival, with shape (masses, mixings, particles).
    :param momenta: The ParticleBatch object containing the momenta information.
    :param index_mass: Index for the desired mass in the global mass_hnl array.
    :param index_mixing: Index for the desired mixing in the global mixing array.
    :param title: Title of the plot.
    :param savename: If provided, the plot will be saved to this path.
    """
    vertices = r_labs[index_mass, index_mixing, :, :] * nat_to_m()
    survived = survival_bool[index_mass, index_mixing, :]

    survived_vertices = vertices[survived]
    failed_vertices = vertices[~survived]

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Plot survived vertices, with legendgroup to group related elements in the legend
    survived_scatter = go.Scatter3d(x=survived_vertices[:, 0], y=survived_vertices[:, 1], z=survived_vertices[:, 2], mode='markers', marker=dict(size=2, color='blue'), name='Survived', legendgroup='survived')
    fig.add_trace(survived_scatter)

    # Plot failed vertices, set to be hidden by default and grouped in legend
    failed_scatter = go.Scatter3d(x=failed_vertices[:, 0], y=failed_vertices[:, 1], z=failed_vertices[:, 2], mode='markers', marker=dict(size=2, color='red'), name='Failed', visible='legendonly', legendgroup='failed')
    fig.add_trace(failed_scatter)

    trajectory_length = 0.5  # Adjust this value to control the length of the trajectories
    for particle_type in ['displaced_minus', 'displaced_plus']:
        momenta.mass(mass_hnl[index_mass]).particle(particle_type)
        if survived.sum() > 0:  # Check if there are survived particles to plot
            px = momenta.px()[survived]
            py = momenta.py()[survived]
            pz = momenta.pz()[survived]

            norms = np.sqrt(px**2 + py**2 + pz**2)
            unit_vectors = np.vstack([px/norms, py/norms, pz/norms]).T

            end_points = survived_vertices + trajectory_length * unit_vectors

            for start, end in zip(survived_vertices, end_points):
                fig.add_trace(go.Scatter3d(
                    x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                    mode='lines',
                    line=dict(color='green' if particle_type == 'displaced_minus' else 'purple', width=3),
                    legendgroup='survived',  # Group with survived
                    showlegend=False  # Do not show these lines individually in the legend
                ))

    fig.update_layout(title=title,
                  scene=dict(
                      xaxis_title='X',
                      yaxis_title='Y',
                      zaxis_title='Z',
                      camera=dict(
                          up=dict(x=0, y=0, z=1),  # Ensures Z is up, but might need adjustments
                          center=dict(x=0, y=0, z=0),  # Center of view
                          eye=dict(x=2, y=2, z=0.1)  # Adjust for a more extreme perspective
                      )
                  ))
    fig.show()


def find_closest_indices(target_mass, target_mixing):
    """
    Finds the indices of the closest mass and mixing values to the target values.

    :param target_mass: The target mass value.
    :param target_mixing: The target mixing value.
    :param mass_hnl: The array of HNL masses.
    :param mixing: The array of mixing values.
    :return: A tuple of indices (mass_index, mixing_index).
    """

    mass_hnl_array = np.array(mass_hnl)
    mixing_array = np.array(mixing)

    # Calculate the absolute difference between the target values and the available values
    mass_diff = np.abs(mass_hnl_array - target_mass)
    mixing_diff = np.abs(mixing_array - target_mixing)

    # Find the indices of the minimum differences
    mass_index = np.argmin(mass_diff)
    mixing_index = np.argmin(mixing_diff)

    return mass_index, mixing_index

def find_best_survival_indices(survival_dv):
    """
    Find the indices of the mass and mixing scenario with the highest survival rate.

    :param survival_dv: A boolean array with shape (masses, mixings, particles) indicating survival.
    :return: A tuple containing the indices of the mass and mixing scenario with the highest survival rate.
    """
    # Calculate the survival rate for each mass and mixing combination
    survival_rates = np.mean(survival_dv, axis=2)  # Average over the particle dimension

    # Find the index of the highest survival rate
    index_mass, index_mixing = np.unravel_index(np.argmax(survival_rates), survival_rates.shape)

    return index_mass, index_mixing

def plot_production_heatmap(production, title='Production Rates', savename=''):
    # Create a meshgrid for interpolation
    mass_grid, mixing_grid = np.meshgrid(np.linspace(min(mass_hnl), max(mass_hnl), 100),
                                         np.logspace(np.log10(min(mixing)), np.log10(max(mixing)), 100))
    
    # Flatten the mass_hnl and mixing arrays for griddata input
    points = np.array([np.repeat(mass_hnl, len(mixing)), np.tile(mixing, len(mass_hnl))]).T
    values = production.flatten()
    
    # Interpolate the production values onto the meshgrid
    production_grid = griddata(points, values, (mass_grid, mixing_grid), method='linear')
    
    plt.figure(figsize=(10, 6))
    
    # Define levels for contourf based on the range of production values
    levels = np.linspace(values.min(), values.max(), 100)
    
    # Create a filled contour plot with interpolated production data
    contour_filled = plt.contourf(mass_grid, mixing_grid, production_grid, levels=levels, cmap='viridis', extend='both')
    plt.colorbar(contour_filled, label='Production Rate')
    
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('HNL Mass (GeV)')
    plt.ylabel('Mixing')
    plt.title(title)
    plt.grid(alpha=0.25)
    
    # Highlight specific regions or levels if needed
    # plt.contour(mass_grid, mixing_grid, production_grid, levels=[threshold], colors='red', linewidths=2)
    
    if savename:
        save_plot(savename)  # Assuming save_plot is a custom function for saving the plot
    
    plt.show()




def plotting(momenta, batch, production_arrays, arrays):
    print('------------------------------- Plotting ----------------------------')
    # Loading of arrays supplied from main:
    survival_dv_displaced, survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_deltaR_displaced, r_lab, lifetimes_rest, lorentz_factors = arrays
    production_nocuts, production_allcuts, production_pT, production_rap, production_invmass, production_dv, production__pT_rap, production__pT_rap_invmass = production_arrays
    
    current_directory = os.getcwd()
    data_path         = os.path.join(current_directory,'data', data_folder) # Data folder path
    plot_path         = os.path.join(data_path, 'Plots')

    if delete_png_before_use:
        delete_png_files(plot_path)

    # Plots:
    if preliminary_plots:
        dv_plot_data = [
            {'data': abs(ParticleBatch(momenta).mass('7 GeV').particle('displaced_minus').pT()), 'label': '$\\mu^-$','linestyle': '-'},
            {'data': abs(ParticleBatch(momenta).mass('7 GeV').particle('boson').pT()), 'label': '$W^\\pm$', 'linestyle': '--'},
            {'data': abs(ParticleBatch(momenta).mass('7 GeV').particle('hnl').pT()), 'label': '$N$','linestyle': '-'},
            {'data': abs(ParticleBatch(momenta).mass('7 GeV').particle('prompt').pT()), 'label': '$\\tau$','linestyle': '--'},
        ]
        plot_histograms(
            data_list=dv_plot_data,
            title='Transverse momentum distribution from 7 GeV HNLs',
            x_label='$p_T$ (GeV)',
            y_label='Frequency',
            savename='pT_distribution'
        )



        eta_plot_data = [
            {'data': ParticleBatch(momenta).mass('7 GeV').particle('displaced_minus').eta(), 'label': '$\\mu^-$','linestyle': '-'},
            {'data': ParticleBatch(momenta).mass('7 GeV').particle('boson').eta(), 'label': '$W^\\pm$', 'linestyle': '--'},
            {'data': ParticleBatch(momenta).mass('7 GeV').particle('hnl').eta(), 'label': '$N$','linestyle': '-'},
            {'data': ParticleBatch(momenta).mass('7 GeV').particle('prompt').eta(), 'label': '$\\tau$','linestyle': '--'},
        ]
        plot_histograms(
            data_list=eta_plot_data,
            title='Pseudorapidity distribution from 7 GeV HNLs',
            x_label='$\\eta$',
            y_label='Frequency',
            savename='pseudorapidity_distribution'
        )

        rapidity_mask = ParticleBatch(momenta).mass('7 GeV').particle('displaced_minus').cut_rap()
        eta_plot_data = [
            {'data': ParticleBatch(momenta).mass('7 GeV').particle('displaced_minus').eta(), 'label': '$\\mu^-$','linestyle': '-'},
            {'data': ParticleBatch(momenta).mass('7 GeV').particle('displaced_minus').apply_mask(rapidity_mask).eta(), 'label': '$\\mu^-$ ($\\eta$-cut)','linestyle': '--'},
        ]
        plot_histograms(
            data_list=eta_plot_data,
            title='Pseudorapidity distribution from 7 GeV HNLs',
            x_label='$\\eta$',
            y_label='Frequency',
            savename='pseudorapidity_distribution_with_cut'
        )

    # Decay vertex plots (enable/disable):
    if plots_decay_stats:
        index_mass, index_mixing = find_closest_indices(target_mass=3, target_mixing=1e-6)
        lifetime_data =[{'data': lifetimes_rest[index_mass,index_mixing], 'label': '$\\tau_N$','linestyle': '-'}]
        plot_histograms(
            data_list=lifetime_data, 
            title=f'Distribution of HNL lifetimes in rest frame for $M_N=${mass_hnl[index_mass]} and $\Theta_\\tau = $ {mixing[index_mixing]}',
            x_label='1/GeV',
            y_label='Frequency',
            savename='HNL_proper_time'
        )

        lifetime_data =[{'data': lifetimes_rest[index_mass,index_mixing]*lorentz_factors[index_mass,index_mixing]*nat_to_m(), 'label': '$\\tau_N$','linestyle': '-'}]
        plot_histograms(
            data_list=lifetime_data, 
            title=f'Distribution of HNL lab decay distances for $M_N=${mass_hnl[index_mass]} and $\Theta_\\tau = $ {mixing[index_mixing]}',
            x_label='m',
            y_label='Frequency',
            savename='HNL_decay_distance_lab'
        )

        lorentz_factor_data =[{'data': lorentz_factors[index_mass,index_mixing], 'label': '$\\gamma_N$','linestyle': '-'}]
        plot_histograms(
            data_list=lorentz_factor_data,
            title=f'Distribution of HNL Lorentz factors in rest frame for $M_N=${mass_hnl[index_mass]} and $\Theta_\\tau = $ {mixing[index_mixing]}',
            x_label='$\gamma$',
            y_label='Frequency',
            savename='Lorentz_factor'
        )

    index_mass, index_mixing = find_closest_indices(target_mass=6.5, target_mixing=4e-5)
    index_mass_best, index_mixing_best = find_best_survival_indices(survival_dv_displaced)
    if dynamic_plot:
        plot_decay_vertices_with_trajectories(r_lab, survival_dv_displaced,batch, index_mass_best,index_mixing_best, title=f'Surviving Decay Vertices in 3D Space for $M_N=${mass_hnl[index_mass_best]}, $\\Theta_\\tau^2 ≈$ {mixing[index_mixing_best]}')
    
    # Parameter space and production plots:
    plot_parameter_space_region(production_allcuts)
    plot_parameter_space_regions(production_nocuts, production_pT, production__pT_rap, production__pT_rap_invmass, production_allcuts, labels=['no cuts', '$p_T$-cut', '($p_T \\cdot \\eta$)-cut', '($p_T \\cdot \\eta \\cdot m_0$)-cut', '($p_T \\cdot \\eta \\cdot m_0 \\cdot \Delta_R \\cdot DV$)-cut'], colors=['red', 'blue', 'green', 'purple', 'black'], smooth=False, sigma=1) 

    plot_production_heatmap(production_allcuts, title='Production Rates (all cuts)', savename='production_allcuts')
    plot_production_heatmap(production_nocuts, title='Production Rates (no cuts)', savename='production_nocuts')
    plot_production_heatmap(production_dv, title='Production Rates (DV cut)', savename='production_dvcut')
    plot_production_heatmap(production_invmass, title='Production Rates (Invariant mass cut)', savename='production_invmass')
    return 0
