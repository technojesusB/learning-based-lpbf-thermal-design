# src/lpbf/constants.py

# Stefan-Boltzmann constant [W / (m^2 K^4)]
SIGMA_SB = 5.670374419e-8

# Absolute Zero in Celsius (for K conversion if needed, though we stick to K)
ABSOLUTE_ZERO_C = -273.15

# Liquidus threshold in normalised temperature space.
# T_norm = (T_phys - T_ambient) / T_ref, with T_ambient=300 K, T_ref=2000 K.
# SS316L liquidus ≈ 1723 K → T_norm ≈ (1723-300)/2000 = 0.7115.
# Conservative value 0.6 used to account for training-set temperature range.
T_LIQUIDUS_NORM: float = 0.6
