# W2HNL
## Overview
This Python program is developed for signal estimation of feebly interacting massive particles (FIMPs) for the ATLAS experiment at the Large Hadron Collider (LHC) with special focus on heavy neutral leptons decaying macroscopically to di-leptons.

## Features

- **ATLAS Geometry Integration**: Simulates the ATLAS detector geometry to identify potential displaced vertices.
- **HNL Decay Modeling**: Includes models for HNL decay into di-lepton pairs, considering various decay channels and lifetimes.
- **Event Reconstruction**: Efficiently reconstructs events from simulated data and computing favorable parameter spaces for feasability studies.
- **Data Analysis Tools**: Provides tools for data filtering, analysis, and visualization to identify significant signals of HNL decay.
- **Customizable Parameters**: Allows users to adjust key parameters like HNL mass, mixing angles and experiemntal parameters (integrated luminosity, allowable decay volume, track reconstruction efficiencies and various pT/eta/invariant mass cuts).

## Installation

```bash
git clone https://github.com/edtireli/W2HNL.git
cd W2HNL
pip install -r requirements.txt
