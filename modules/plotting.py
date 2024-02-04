import matplotlib.pyplot as plt


def momentum_distribution(batch, batch_mass='1 GeV'):
    bin_number = 40
    plt.title(f'Transverse momentum distribution for {batch_mass} HNL events')
    plt.ylabel('Frequency')
    plt.xlabel('Transverse momentum (GeV)')
    plt.hist(abs(batch.mass(batch_mass).momentum('boson').pT()),           histtype='step', bins=bin_number, label ='boson', linestyle='--')
    plt.hist(abs(batch.mass(batch_mass).momentum('hnl').pT()),             histtype='step', bins=bin_number, label ='hnl', linestyle='-')
    plt.hist(abs(batch.mass(batch_mass).momentum('prompt').pT()),          histtype='step', bins=bin_number, label ='tau', linestyle='--')
    plt.hist(abs(batch.mass(batch_mass).momentum('displaced_minus').pT()), histtype='step', bins=bin_number, label ='mu-', linestyle='-')
    plt.hist(abs(batch.mass(batch_mass).momentum('displaced_plus').pT()),  histtype='step', bins=bin_number, label ='mu+', linestyle='-')
    plt.hist(abs(batch.mass(batch_mass).momentum('neutrino').pT()),        histtype='step', bins=bin_number, label ='vt', linestyle='dotted')
    plt.yscale('log')
    plt.legend()
    plt.show()

def momentum_cut_comparison(batch, batch_mass='1 GeV', particle_name='displaced_minus', pT_cut_condition = '5 GeV'):
    bin_number = 40
    unfiltered_events = abs(batch.mass(batch_mass).momentum(particle_name).pT())
    filtered_events   = abs(batch.mass(batch_mass).momentum(particle_name).cut_pT(pT_cut_condition).pT())
    print(f'Survival probability = {len(filtered_events)/len(unfiltered_events)}')

    plt.title(f'Transverse momentum distribution for {batch_mass} HNL events')
    plt.ylabel('Frequency')
    plt.xlabel('Transverse momentum (GeV)')
    plt.hist(unfiltered_events, histtype='step', bins=bin_number, label ='Unfiltered events', linestyle='-')
    plt.hist(filtered_events, histtype='step', bins=bin_number, label ='Filtered events', linestyle='-')
    plt.yscale('log')
    plt.legend()
    plt.show()

def plotting(batch):
    momentum_distribution(batch,'1 GeV')
    momentum_distribution(batch,'10 GeV')

    momentum_cut_comparison(batch, '1 GeV', 'displaced_minus', pT_cut_condition='5 GeV')
    momentum_cut_comparison(batch, '10 GeV', 'displaced_minus', pT_cut_condition='5 GeV')
    return 0