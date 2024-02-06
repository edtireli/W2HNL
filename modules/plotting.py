import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from modules.data_processing import *
from parameters.data_parameters import *
from parameters.experimental_parameters import *
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

def momentum_distribution(batch, batch_mass='1 GeV'):
    bin_number = 40
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


def plot_histograms(data_list, title, x_label, y_label):
    """
    Plots a series of histograms with dynamic inputs.

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
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")

    for entry in data_list:
        plt.hist(entry['data'], histtype='step', bins=40, label=entry['label'], linestyle=entry.get('linestyle', '-'))

    plt.legend()
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
    plt.xlabel('HNL Mass, $M_N$ (GeV)')
    plt.ylabel('Mixing, $\\Theta_{\\tau}^2$')
    plt.grid(alpha=0.25)
    plt.title(f'HNL production parameter space')

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
    plt.xlabel('HNL Mass, $M_N$ (GeV)')
    plt.ylabel('Mixing, $\\Theta_{\\tau}^2$')
    plt.title('HNL Production Parameter Space')
    plt.grid(alpha=0.25)
    plt.show()

def mean_from_2d_survival(arr):
    temp_array = [np.mean(i) for i in arr]
    return temp_array

def plotting(momenta, batch, production_arrays, arrays):
    print('------------------------------- Plotting ----------------------------')

    survival_dv_displaced, survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_deltaR_displaced = arrays
    production_allcuts, production_pT, production_rap, production_invmass, production__pT_rap, production__pT_rap_invmass = production_arrays
    
    dv_plot_data = [
        {'data': abs(ParticleBatch(momenta).mass('7 GeV').particle('displaced_minus').pT()), 'label': '$\\mu^-$','linestyle': '-'},
        {'data': abs(ParticleBatch(momenta).mass('7 GeV').particle('boson').pT()), 'label': '$W^\\pm$', 'linestyle': '--'},
        {'data': abs(ParticleBatch(momenta).mass('7 GeV').particle('hnl').pT()), 'label': '$N$','linestyle': '-'},
        {'data': abs(ParticleBatch(momenta).mass('7 GeV').particle('prompt').pT()), 'label': '$\\tau$','linestyle': '--'},
    ]

    plot_histograms(
        data_list=dv_plot_data,
        title='Transverse momentum distribution from 7 GeV HNLs',
        x_label='pT (GeV)',
        y_label='Frequency'
    )

    #plot_survival_3d(survival_dv_displaced, 'DV cut')
    #plot_survival_2d([survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_deltaR_displaced], ['$p_T$ cut', '$\\eta$ cut', 'm_0 cut', '$\Delta_R$ cut'])


    #plot_survival_3d(combined_survival, '(DV $\cdot$ $\\eta \cdot p_T \cdot \Delta_R \cdot m_0$) cut')
    plot_parameter_space_region(production_allcuts)
    plot_parameter_space_regions(production_pT, production__pT_rap, production__pT_rap_invmass, production_allcuts, labels=['$p_T$-cut', '($p_T \\cdot \\eta$)-cut', '($p_T \\cdot \\eta \\cdot m_0$)-cut', '($p_T \\cdot \\eta \\cdot m_0 \\cdot \Delta_R \\cdot DV$)-cut'], colors=['red', 'blue', 'green', 'black'], smooth=True, sigma=1) 

    return 0