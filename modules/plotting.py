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
from matplotlib import cm


#Plot triggers (enable/disable for extra plots):

# Delete all stored images in Plots subfolder before saving new
delete_png_before_use = False

# Simple pT and rapidity plots
preliminary_plots = True

# Dynamic plot (Web hosted 3d lab frame decay vertex plot with daughters as arrows)
dynamic_plot = True

# Decay vertex statistics for specific mass/mix HNLs (configure below by plot)
plots_decay_stats = False

# Heat maps of production after various cuts
plot_heatmaps = True 

# Plot survival heatmaps
survival_plots = True
survival_plots_analysis = False # Extra analysis vs data plots (time consuming)
levels_heatmap = 10 #The number of levels on the contour plots

# Plot production heatmaps
production_plots = True

# Functions:

def on_key_press(event):
    """Close the plot when the ESC key is pressed."""
    if event.key == 'escape':
        plt.close(event.canvas.figure)

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

    # Connect the key press event to the handler
    plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)

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

    
    from matplotlib.ticker import AutoMinorLocator
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(10)) 
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

def save_contour_data(contour_data, name):
    current_directory = os.getcwd()
    data_path = os.path.join(current_directory, 'data', data_folder)
    contour_data_path = os.path.join(data_path, 'Plots', 'Plot data', f'{name}.pkl')
    os.makedirs(os.path.join(data_path, 'Plots', 'Plot data'), exist_ok=True)  # Ensure the directory exists

    # Save the contour data using pickle
    with open(contour_data_path, 'wb') as file:
        pickle.dump(contour_data, file)
    print(f"Contour data saved to {contour_data_path}")

def save_survival_data(survival_data, name):
    current_directory = os.getcwd()
    data_path = os.path.join(current_directory, 'data', data_folder)
    survival_data_path = os.path.join(data_path, 'Plots', 'Plot data', f'{name}.pkl')
    os.makedirs(os.path.join(data_path, 'Plots', 'Plot data'), exist_ok=True)  # Ensure the directory exists

    # Save the contour data using pickle
    with open(survival_data_path, 'wb') as file:
        pickle.dump(survival_data, file)
    print(f"Survival data saved to {survival_data_path}")    

def plot_parameter_space_region(production_allcuts, title='', savename=''):
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(mass_hnl, mixing)

    cmap = plt.get_cmap('viridis')
    mesh = plt.pcolormesh(X, Y, production_allcuts.T, cmap=cmap, shading='auto')
    plt.colorbar(mesh, label='Production')

    # Draw contours around regions where production exceeds the minimum thresholds
    if production_minimum is not None:
        mask_min = production_allcuts.T >= production_minimum  # Transpose to align dimensions
        plt.contour(X, Y, mask_min.astype(int), levels=[0.5], colors='red', linewidths=1.5)

    if production_minimum_secondary is not None:
        mask_min_2 = production_allcuts.T >= production_minimum_secondary
        plt.contour(X, Y, mask_min_2.astype(int), levels=[0.5], colors='blue', linewidths=1.5)

    plt.xscale('linear')
    plt.yscale('log')
    plt.ylim(min(mixing), max(mixing))
    plt.xlabel('HNL Mass, $M_N$ (GeV)', size=12)
    plt.ylabel('Mixing, $\\Theta_{\\tau}^2$', size=12)
    plt.title(title)
    plt.grid(alpha=0.25)

    # Connect the key press event to the handler
    plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)

    # Save plot if savename is provided
    if savename:
        save_plot(savename)

def save_data(mass_grid, mixing_grid, production_grid, filename):
    current_directory = os.getcwd()
    data_path = os.path.join(current_directory, 'data', data_folder)
    file_path = os.path.join(data_path, 'Plots', 'Plot data', filename)
    os.makedirs(os.path.join(data_path, 'Plots', 'Plot data'), exist_ok=True)  # Ensure the directory exists
    with open(file_path, 'wb') as f:
        np.save(f, mass_grid)
        np.save(f, mixing_grid)
        np.save(f, production_grid)
        print(f'Production data saved to {filename}')

def load_data(filename):
    with open(filename, 'rb') as f:
        mass_grid = np.load(f)
        mixing_grid = np.load(f)
        production_grid = np.load(f)
    return mass_grid, mixing_grid, production_grid

def plot_from_saved_data(filename):
    mass_grid, mixing_grid, production_grid = load_data(filename)
    plt.figure(figsize=(10, 6))
    levels = np.linspace(0, production_grid.max(), 5)
    contour_filled = plt.contourf(mass_grid, mixing_grid, production_grid, levels=levels, extend='max', cmap='Greens')
    plt.colorbar(contour_filled, label='Production')
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('HNL Mass, $M_N$ (GeV)', size=12)
    plt.ylabel('Mixing, $\\Theta_{\\tau}^2$', size=12)
    plt.grid(alpha=0.25)
    plt.show()
    
def plot_parameter_space_regions(*production_arrays, labels=None, colors=None, smooth=False, sigma=1, savename):
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
    
    # Connect the key press event to the handler
    plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)

    plt.grid(alpha=0.25)
    save_plot(savename)
    plt.show()

def plot_survival_fractions_simple(survival_arrays, labels, title, savename):
    """
    Plot survival fractions for various cut conditions with simplified inputs.

    :param survival_arrays: List of arrays containing survival fractions.
    :param labels: List of labels for each array's plot.
    :param title: Title for the plot.
    :param savename: Filename for saving the plot.
    """
    plt.figure(figsize=(10, 6))

    colors = ['red', 'black', 'blue', 'green']
    linstyles = ['-','--','dotted','-.']

    for i, value in enumerate(survival_arrays):
        mean_survival_fractions = value.mean(axis=1)
        plt.plot(mass_hnl, mean_survival_fractions, color=colors[i], label=labels[i], linewidth=2, alpha=0.75, linestyle=linstyles[i])

    plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)

    plt.legend(loc='lower right', frameon=True)
    plt.xlabel('HNL Mass, $M_N$ (GeV)', size=12)
    plt.ylabel('Survival Fraction', size=12)
    plt.title(title)
    plt.xticks(mass_hnl)
    plt.yticks([i*1e-1 for i in range(0,11)])
    plt.grid(alpha=0.25)
    save_plot(savename)
    plt.show()

def plot_survival_parameter_space_regions(survival_fraction, labels=None, colors=None, title='', savename='', plot_mass_mixing_lines=False):
    """
    Plot survival fraction on the parameter space with contours.

    :param survival_fraction: Survival fraction array with shape (masses, mixings).
    :param labels: Optional list of labels for the contour.
    :param colors: Optional color for the contour line.
    :param smooth: Apply Gaussian smoothing to the survival fraction.
    :param sigma: Standard deviation for Gaussian kernel if smoothing is applied.
    """
    mass_grid, mixing_grid = np.meshgrid(
        np.linspace(min(mass_hnl), max(mass_hnl), 100),
        np.logspace(np.log10(min(mixing)), np.log10(max(mixing)), 100)
    )

    plt.figure(figsize=(10, 6))

    if labels is None:
        labels = 'Survival Fraction'
    if colors is None:
        colors = 'blue'

    # Interpolate survival fraction onto the grid
    survival_grid = griddata(
        (np.repeat(mass_hnl, len(mixing)), np.tile(mixing, len(mass_hnl))),
        survival_fraction.flatten(),
        (mass_grid, mixing_grid),
        method='linear'
    )

    plt.contourf(
        mass_grid, mixing_grid, survival_grid,
        levels=np.linspace(survival_fraction.min(), survival_fraction.max(), levels_heatmap),
        cmap=plt.cm.viridis
    )
    plt.colorbar(label='Survival Fraction')
    

    constants=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3,1e4,1e5]

    if plot_mass_mixing_lines:
        for C in constants:
            mass_range = np.linspace(min(mass_hnl), max(mass_hnl), 500)
            mixing_for_constant = (C / mass_range**6)
            plt.plot(mass_range, mixing_for_constant, '--', color='red', label=f'C={C:.1e}', alpha=0.4)

            label_index = len(mass_range) // 2  # Midpoint
            label_y_position = mixing_for_constant[label_index] * 0.5
            if C > 1e-4 and C < 1e4:
                plt.text(mass_range[label_index], label_y_position, 
                            f'$c\\tau\\gamma = {C:.1e}$', color='red', fontsize=9,
                            ha='center', va='bottom', rotation=-22, alpha=0.4)

    plt.plot([], [], color=colors, linewidth=2, linestyle='-', label=labels)

    plt.xscale('linear')
    plt.yscale('log')
    plt.ylim(1e-8, 1e0)
    plt.xlabel('HNL Mass, $M_N$ (GeV)', size=12)
    plt.ylabel('Mixing, $\\Theta_{\\tau}^2$', size=12)
    plt.title(title)
    
    # Connect the key press event to the handler
    plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)

    plt.grid(alpha=0.25)
    save_plot(savename)
    save_survival_data(survival_fraction, savename)
    plt.show()

def plot_survival_parameter_space_regions_nointerpolation(survival_fraction, labels=None, colors=None, title='', savename='', plot_mass_mixing_lines=False, gammas=None):
    """
    Plot survival fraction on the parameter space directly using the actual survival values.

    :param survival_fraction: Survival fraction array with shape (masses, mixings).
    :param labels: Optional list of labels for the plot.
    :param colors: Optional color for the plot.
    :param title: Title of the plot.
    :param savename: Filename to save the plot.
    :param plot_mass_mixing_lines: Boolean to plot mass-mixing lines.
    """
    plt.figure(figsize=(10, 6))

    # Assuming mass_hnl and mixing are global variables defining the axes.
    X, Y = np.meshgrid(mass_hnl, mixing)
    
    # Use pcolormesh to plot the survival fraction directly.
    cmap = plt.get_cmap('viridis')
    mesh = plt.pcolormesh(X, Y, survival_fraction.T, cmap=cmap, shading='auto')  # Transpose to align dimensions

    plt.colorbar(mesh, label='Survival Fraction')

    if plot_mass_mixing_lines:
        constants = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]
        c = light_speed() 

        for C in constants:
            if C <= 1e-4 or C >= 1e5:
                continue 

            tau_values = [HNL(mass, [0,0,1], False).computeNLifetime() for mass in mass_hnl]
            mixing_values_for_C = np.array([C / (tau * c * gammas[index]) for index, tau in enumerate(tau_values)])
            reciprocal_mixing_values = 1 / mixing_values_for_C

            plt.plot(mass_hnl, reciprocal_mixing_values, '--', color='red', label=f'C={1/C:.1e}', alpha=0.4)

            label_index = len(mass_hnl) // 2
            if 1/C <= 1e-4 or 1/C >= 1e5:
                continue 

            # Format the C value as a power of ten for the label
            exponent = int(np.log10(1/C))
            if 1/C == 1:
                C_label = '1'
            elif exponent < 0:
                C_label = f'10^{{{abs(exponent)}}}'
            else:
                C_label = f'10^{{-{exponent}}}'
            plt.text(mass_hnl[label_index], reciprocal_mixing_values[label_index], f'$c\\tau\\gamma = {C_label}$ m', color='red', fontsize=9,
                    ha='center', va='bottom', rotation=-15, alpha=0.4)


    plt.xscale('linear')
    plt.yscale('log')
    plt.ylim(min(mixing), max(mixing))
    plt.xlabel('HNL Mass, $M_N$ (GeV)', size=12)
    plt.ylabel('Mixing, $\\Theta_{\\tau}^2$', size=12)
    plt.title(title)
    plt.grid(alpha=0.25)

    # Connect the key press event to the handler
    plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)

    # Save plot if savename is provided
    if savename:
        save_plot(savename)
    
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

def save_array(array, name=''):
    current_directory = os.getcwd()                                         # Current path
    data_path         = os.path.join(current_directory,'data', data_folder) # Data folder path
    array_path         = os.path.join(data_path, 'Plots', 'Plot data', f'{name}.npy')
    os.makedirs(os.path.join(data_path, 'Plots'), exist_ok=True) # Making directory if not already exists

    np.save(array_path, array)

def expand_and_copy_array(input_array):
    """
    Expand an array of shape (masses, particles) to shape (masses, mixings, particles)
    by copying the input array for each mixing.

    :param input_array: Array of shape (masses, particles).
    :param target_mixings: Number of mixings to expand to.
    :return: Expanded array of shape (masses, mixings, particles).
    """
    # Ensure the input_array is at least 3D (masses, 1, particles)
    input_array_expanded = np.expand_dims(input_array, axis=1)
    
    # Repeat the array across the mixing dimension
    expanded_array = np.tile(input_array_expanded, (1, len(mixing), 1))
    
    return expanded_array

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

def compute_analytic_survival(gammas):
    survival_array = np.zeros((len(mass_hnl), len(mixing)))
    for i, m in enumerate(mass_hnl):
        for j, mix in enumerate(mixing):
            angles = [0, 0, mix]
            tau = HNL(m, angles, False).computeNLifetime()
            P_values = []
            for gamma in gammas[i,j]:
                P = np.exp(-unit_converter(r_min) / (light_speed() * tau * gamma)) - np.exp(-unit_converter(r_max_l) / (light_speed() * tau * gamma))
                P_values.append(P)
            survival_array[i, j] = np.mean(P_values)
    save_array(survival_array, name='survival_dv_array_analytic')
    return survival_array

def compute_analytic_survival_averaged(gamma_avg):
        survival_array = np.zeros((len(mass_hnl), len(mixing)))
        for i, m in enumerate(mass_hnl):
            for j, mix in enumerate(mixing):
                angles = [0, 0, mix]
                tau = HNL(m, angles, False).computeNLifetime()
                gamma = gamma_avg[i]
                P = np.exp(-(unit_converter(r_min)) / (light_speed() * tau * gamma)) - np.exp(-(unit_converter(r_max_l)) / (light_speed() * tau * gamma))
                survival_fraction = P
                survival_array[i, j] = survival_fraction
        save_array(survival_array, name='survival_dv_array_analytic_avg')
        return survival_array

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


def plot_production_heatmap(production, title='Production Rates', savename='', save_grid=False):
    # Create a meshgrid for interpolation
    mass_grid, mixing_grid = np.meshgrid(np.linspace(min(mass_hnl), max(mass_hnl), 100),
                                         np.logspace(np.log10(min(mixing)), np.log10(max(mixing)), 100))
    
    # Flatten the mass_hnl and mixing arrays for griddata input
    points = np.array([np.repeat(mass_hnl, len(mixing)), np.tile(mixing, len(mass_hnl))]).T
    values = production.flatten()
    
    # Interpolate the production values onto the meshgrid
    production_grid = griddata(points, values, (mass_grid, mixing_grid), method='linear')
    if save_grid:
        current_directory = os.getcwd()
        data_path = os.path.join(current_directory, 'data', data_folder)
        save_path = os.path.join(data_path, 'Plots', 'Plot data')
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        
        data_to_save = (mass_grid, mixing_grid, production_grid, values)
        with open(os.path.join(save_path, f'{savename}.pkl'), 'wb') as file:
            pickle.dump(data_to_save, file)
        print(f"Data saved to {os.path.join(save_path, f'{savename}.pkl')}")

    plt.figure(figsize=(10, 6))
    
    # Define levels for contourf based on the range of production values
    levels = np.linspace(values.min(), values.max(), 100)
    
    # Create a filled contour plot with interpolated production data
    contour_filled = plt.contourf(mass_grid, mixing_grid, production_grid, levels=levels, cmap='viridis', extend='both')
    plt.colorbar(contour_filled, label='Production Rate')
    if save_grid==True:
        constants = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]  

        # Plot lines for each constant
        for C in constants:
            mass_range = np.linspace(min(mass_hnl), max(mass_hnl), 500)
            mixing_for_constant = np.sqrt(C / mass_range**6)
            plt.plot(mass_range, mixing_for_constant, '--', color='r', label=f'C={C:.1e}', alpha=0.5)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('HNL Mass (GeV)')
    plt.ylabel('Mixing')
    plt.title(title)
    
    # Connect the key press event to the handler
    plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)

    plt.grid(alpha=0.25)
    
    # Highlight specific regions or levels if needed
    # plt.contour(mass_grid, mixing_grid, production_grid, levels=[threshold], colors='red', linewidths=2)
    
    if savename:
        save_plot(savename)  
    
    plt.show()


def calculate_survival_fraction(survival_array):
    """
    Calculate the survival fraction for each mass and mixing combination.

    :param survival_array: Array of shape (masses, mixings, particles) with boolean values.
    :return: Array of shape (masses, mixings) with survival fractions.
    """
    # Calculate the sum of True values (survived) along the particle dimension
    survived = np.sum(survival_array, axis=-1)
    # Calculate the total number of particles
    total_particles = survival_array.shape[-1]
    # Calculate the survival fraction
    survival_fraction = survived / total_particles

    return survival_fraction

def print_dashes(text, char='-'):
    width = shutil.get_terminal_size().columns
    side = (width - len(text) - 2) // 2
    print(f"{char * side} {text} {char * (width - side - len(text) - 2)}")


def plotting(momenta, batch, production_arrays, arrays):
    print_dashes('Plotting')
    # Loading of arrays supplied from main:
    if not large_data:
        survival_dv_displaced, survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_deltaR_displaced, r_lab, lifetimes_rest, lorentz_factors = arrays
    else:
        survival_dv_displaced, survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_deltaR_displaced = arrays

    production_nocuts, production_allcuts, production_pT, production_rap, production_invmass, production_dv, production__pT_rap, production__pT_rap_invmass = production_arrays
    save_array(production_allcuts, f'production_allcuts_{luminosity}')
    if not large_data:
        save_array(lifetimes_rest, name='lifetime_rest_data')
        average_lorentz_factors = np.mean(lorentz_factors, axis=(1, 2))

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

    # Decay vertex plots (enable/disable):
    if not large_data:
        if plots_decay_stats:
            index_mass, index_mixing = find_closest_indices(target_mass=6, target_mixing=1e-6)
            lifetime_data =[{'data': lifetimes_rest[index_mass,index_mixing], 'label': '$\\tau_N$','linestyle': '-'}]
            plot_histograms(
                data_list=lifetime_data, 
                title=f'Distribution of HNL lifetimes in rest frame for $M_N=${mass_hnl[index_mass]} and $\Theta_\\tau = $ {mixing[index_mixing]}',
                x_label='1/GeV',
                y_label='Frequency',
                savename='HNL_proper_time'
            )

            lifetime_data =[{'data': lifetimes_rest[index_mass,index_mixing]*lorentz_factors[index_mass,index_mixing]*light_speed(), 'label': '$\\tau_N$','linestyle': '-'}]
            plot_histograms(
                data_list=lifetime_data, 
                title=f'Distribution of HNL lab decay distances for $M_N=${mass_hnl[index_mass]} and $\Theta_\\tau = $ {mixing[index_mixing]}',
                x_label='m',
                y_label='Frequency',
                savename='HNL_decay_distance_lab'
            )

            _, index_mixing = find_closest_indices(target_mass=6, target_mixing=1)
            rd_lab_norm = np.linalg.norm(r_lab, axis=-1)
            rd_lab_norm_selected = rd_lab_norm[:, index_mixing]  # Shape: (len(mass_hnl), batch_size)
            lifetimes_rest_selected = lifetimes_rest[:, index_mixing]  # Shape: (len(mass_hnl), batch_size)
            lorentz_factors_selected = lorentz_factors[:, index_mixing]  # Shape: (len(mass_hnl), batch_size)

            normalized_decay_distances = rd_lab_norm_selected / (lifetimes_rest_selected * light_speed() * lorentz_factors_selected)
            average_normalized_decay_distance = np.mean(normalized_decay_distances, axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(mass_hnl, average_normalized_decay_distance, marker='o', linestyle='-', color='blue')
            plt.xlabel('HNL Mass (GeV)')
            plt.ylabel('Average Normalized Decay Distance $\\frac{L}{c\\tau\\gamma}$')
            plt.title('Average Normalized Decay Distance vs. HNL Mass')
            plt.grid(True)
            save_plot('Lctg')
            plt.show()
            
            plt.figure(figsize=(10, 6))
            plt.scatter(mass_hnl, average_lorentz_factors, linestyle='-', color='k', alpha=0.75, s=4)
            plt.xlabel('HNL mass $M_N$ (GeV)')
            plt.ylabel('Average Lorentz factor $\gamma_N$')
            plt.title('Average Lorentz factors as a function of HNL mass')
            plt.grid(True)
            save_plot('lorentz_factors')
            plt.show()


        index_mass, index_mixing = find_closest_indices(target_mass=6.5, target_mixing=4e-5)
        index_mass_best, index_mixing_best = find_best_survival_indices(survival_dv_displaced)
        if dynamic_plot:
            plot_decay_vertices_with_trajectories(r_lab, survival_dv_displaced,batch, index_mass_best,index_mixing_best, title=f'Surviving Decay Vertices in 3D Space for $M_N=${mass_hnl[index_mass_best]}, $\\Theta_\\tau^2 ≈$ {mixing[index_mixing_best]}')
        
    # Survival plots: 
    if survival_plots:
        # DV cut heatmap from data
        survival_dv_fraction = calculate_survival_fraction((survival_dv_displaced))
        plot_survival_parameter_space_regions_nointerpolation(survival_dv_fraction, title='HNL survival (DV cut)', savename='survival_dv', plot_mass_mixing_lines = True, gammas = average_lorentz_factors)
        
        if survival_plots_analysis:
            # DV cut heatmap from analysis (slow)
            survival_dv_analytic = compute_analytic_survival(lorentz_factors)
            plot_survival_parameter_space_regions_nointerpolation(survival_dv_analytic, title='HNL survival (analytic (2) DV cut)', savename='survival_dv_analytic_2', plot_mass_mixing_lines = True, gammas = average_lorentz_factors)
            
            # Difference between data and analytic DV cut survival heatmap
            survival_dv_delta = survival_dv_fraction - survival_dv_analytic
            plot_survival_parameter_space_regions_nointerpolation(survival_dv_delta, title='HNL DV cut survival difference (data method - analytic (2) method)', savename='survival_dv_delta_2', plot_mass_mixing_lines = True, gammas = average_lorentz_factors)
            
            # DV cut heatmap from analysis (slow)
            survival_dv_analytic_avg = compute_analytic_survival_averaged(average_lorentz_factors)
            plot_survival_parameter_space_regions_nointerpolation(survival_dv_analytic_avg, title='HNL survival (analytic (1) DV cut)', savename='survival_dv_analytic_1', plot_mass_mixing_lines = True, gammas = average_lorentz_factors)
            

            # Difference between data and analytic DV cut survival heatmap
            survival_dv_delta_avg = survival_dv_fraction - survival_dv_analytic_avg
            plot_survival_parameter_space_regions_nointerpolation(survival_dv_delta_avg, title='HNL DV cut survival difference (data method - analytic (1) method)', savename='survival_dv_delta_1', plot_mass_mixing_lines = True, gammas = average_lorentz_factors)
            

            # Difference between analytic estimation and averaged analytic estimation
            survival_dv_delta_analysis = survival_dv_analytic - survival_dv_analytic_avg
            plot_survival_parameter_space_regions_nointerpolation(survival_dv_delta_analysis, title='HNL analytic DV cut survival difference (2) - (1)', savename='survival_dv_delta_2-1', plot_mass_mixing_lines = True, gammas = average_lorentz_factors)

            a = False
            if a: 
                index_mass, index_mixing = find_closest_indices(target_mass=9, target_mixing=1e-7)
                average_survival_rate = np.mean(survival_dv_displaced, axis=-1)
                #print('Survival rate for specified mixing: ', average_survival_rate[index_mass, index_mixing])
                #print('mixing: ', mixing[index_mixing])
                #print('mass: ', mass_hnl[index_mass])
                for index_mass in range(len(mass_hnl)):
                    average_lorentz_factor = np.mean(lorentz_factors[index_mass, index_mixing])
                    average_lifetime_rest = np.mean(lifetimes_rest[index_mass, index_mixing])

                    denominator = (average_lifetime_rest * average_lorentz_factor * light_speed())
                    exp_arg_max = -unit_converter(r_max_l) / denominator
                    exp_arg_min = -unit_converter(r_min) / denominator
                    analytic_survival_probability = np.exp(np.minimum(exp_arg_min, 700)) - np.exp(np.minimum(exp_arg_max, 700))
                    # Plotting
                    plt.plot(mass_hnl[index_mass], average_survival_rate[index_mass, index_mixing], 'bo', label='Simulated Survival Rate' if index_mass == 0 else "")
                    plt.plot(mass_hnl[index_mass], analytic_survival_probability, 'rx', label='Analytic Survival Probability' if index_mass == 0 else "")

                plt.xlabel('HNL mass (GeV)')
                plt.ylabel('Survival Rate')
                plt.title(f'Survival Rate Comparison for Mixing $\\Theta_\\tau^2$ = {mixing[index_mixing]}')
                plt.legend()
                save_plot('DV_cut_validation')
                plt.show()
                
        survival_pt_= calculate_survival_fraction(expand_and_copy_array(survival_pT_displaced))
        if invmass_cut_type != 'trivial':
            survival_invmass_ = calculate_survival_fraction(survival_invmass_displaced)
        else:
            survival_invmass_= calculate_survival_fraction(expand_and_copy_array(survival_invmass_displaced))    
        survival_rapidity_= calculate_survival_fraction(expand_and_copy_array(survival_rap_displaced))
        survival_deltaR_= calculate_survival_fraction(expand_and_copy_array(survival_deltaR_displaced))
        

        # Invariant mass survival
        plot_survival_parameter_space_regions_nointerpolation(survival_invmass_, title='HNL survival (invariant mass)', savename='survival_invmass', plot_mass_mixing_lines = False)
        save_array(survival_invmass_, 'survival_invmass')

        # Total survival
        survival_total = survival_dv_fraction * survival_pt_* survival_rapidity_* survival_invmass_* survival_deltaR_
        plot_survival_parameter_space_regions_nointerpolation(survival_total, title='HNL survival (all cuts)', savename='survival_allcuts', plot_mass_mixing_lines = False)
        
        # Old heatmaps that look nice, but dont depend on mixing (so its unecessary having them as heatmaps)
        #plot_survival_parameter_space_regions(calculate_survival_fraction(expand_and_copy_array(survival_pT_displaced)), smooth=False, sigma=1, title='HNL survival (pT cut)', savename='survival_pT')
        #plot_survival_parameter_space_regions(calculate_survival_fraction(expand_and_copy_array(survival_invmass_displaced)), smooth=False, sigma=1, title='HNL survival (invariant mass cut)', savename='survival_invmass')
        #plot_survival_parameter_space_regions(calculate_survival_fraction(expand_and_copy_array(survival_rap_displaced)), smooth=False, sigma=1, title='HNL survival (rapidity cut)', savename='survival_rap')
        
        if invmass_cut_type != 'nontrivial':
            # Heatmap (unecessary)
            #plot_survival_parameter_space_regions(calculate_survival_fraction(expand_and_copy_array(survival_deltaR_displaced)), smooth=False, sigma=1, title='HNL survival ($\\Delta R$ cut)', savename='survival_deltaR')
            
            # Simple survival fraction plots
            plot_survival_fractions_simple([survival_pt_, survival_invmass_, survival_rapidity_, survival_deltaR_], ['$p_T$', '$M_{\\mu\\mu}$', '$\eta$', '$\Delta_R$'], 'Survival fractions', 'survival_fractions_simple')
        else:
            # Heatmap (necessary) because now invariant mass depends on mixing
            plot_survival_fractions_simple([survival_pt_, survival_rapidity_, survival_deltaR_], ['$p_T$', '$\eta$', '$\Delta_R$'], 'Survival fractions', savename='survival_fractions_simple')
            
            # Simple survival fraction plots
            #plot_survival_parameter_space_regions(calculate_survival_fraction(survival_deltaR_displaced), title='HNL survival ($\\Delta R$ cut)', savename='survival_deltaR')

    # Parameter space and production plots:
    plot_parameter_space_region(production_nocuts, title='HNL Production (no cuts)', savename = 'hnl_production_nocuts')  
    save_array(production_nocuts, 'production_nocuts')
    plot_parameter_space_region(production_allcuts, title='HNL Production (all cuts)', savename = 'hnl_production_allcuts')    
    plot_parameter_space_regions(production_nocuts, production_pT, production__pT_rap, production__pT_rap_invmass, production_allcuts, labels=['no cuts', '$p_T$-cut', '($p_T \\cdot \\eta$)-cut', '($p_T \\cdot \\eta \\cdot m_0$)-cut', '($p_T \\cdot \\eta \\cdot m_0 \\cdot \Delta_R \\cdot DV$)-cut'], colors=['red', 'blue', 'green', 'purple', 'black'], smooth=False, sigma=1, savename='hnl_production_parameter_space_multi') 
    if production_plots:
        plot_parameter_space_region(production_invmass, title='HNL Production (invariant mass cut)', savename = 'hnl_production_invmass')
        save_array(production_invmass, 'production_invmass')
        plot_parameter_space_regions(production_nocuts, production_dv, labels=['no cuts', 'DV'], colors=['red', 'black'], smooth=False, sigma=1, savename = 'hnl_production_parameter_space_dv') 

        if plot_heatmaps:
            plot_production_heatmap(production_allcuts, title='Production Rates (all cuts)', savename='production_allcuts')
            plot_production_heatmap(production_dv, title='Production Rates (DV cut)', savename='production_dvcut', save_grid=True)

            # Old and unecessary
            #plot_production_heatmap(production_nocuts, title='Production Rates (no cuts)', savename='production_nocuts')
            #plot_production_heatmap(production_invmass, title='Production Rates (Invariant mass cut)', savename='production_invmass')

    return 0
