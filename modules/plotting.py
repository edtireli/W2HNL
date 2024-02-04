import matplotlib.pyplot as plt


def momentum_distribution(batch, batch_mass='1 GeV'):
    bin_number = 40
    plt.title('Momentum distribution for 1 GeV HNL events')
    plt.ylabel('Frequency')
    plt.xlabel('Momentum (GeV)')
    plt.hist(abs(batch.mass(batch_mass).momentum('boson').pz()),           histtype='step', bins=bin_number, label ='boson', linestyle='--')
    plt.hist(abs(batch.mass(batch_mass).momentum('hnl').pz()),             histtype='step', bins=bin_number, label ='hnl', linestyle='-')
    plt.hist(abs(batch.mass(batch_mass).momentum('prompt').pz()),          histtype='step', bins=bin_number, label ='tau', linestyle='--')
    plt.hist(abs(batch.mass(batch_mass).momentum('displaced_minus').pz()), histtype='step', bins=bin_number, label ='mu-', linestyle='-')
    plt.hist(abs(batch.mass(batch_mass).momentum('displaced_plus').pz()),  histtype='step', bins=bin_number, label ='mu+', linestyle='-')
    plt.hist(abs(batch.mass(batch_mass).momentum('neutrino').pz()),        histtype='step', bins=bin_number, label ='vt', linestyle='dotted')
    plt.yscale('log')
    plt.legend()
    plt.show()

def plotting(batch):
    momentum_distribution(batch,'1 GeV')
    momentum_distribution(batch,'10 GeV')
    return 0