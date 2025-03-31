import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

er_fluence = np.linspace(0, 100, 100)  
mito_fluence = np.linspace(0, 100, 100)  
ER, MITO = np.meshgrid(er_fluence, mito_fluence)


# assume mitchondria has a logistic response
def mito_response(fluence):
    return 0.7 / (1 + np.exp(-0.1 * (fluence - 40)))  # Hill-like sigmoid

# ER has a more gradual response than mitochondria
def er_response(fluence):
    return 0.3 * (1 - np.exp(-0.03 * fluence))  

# somewhat arbitrary crosstalk function
def crosstalk_gain(er_fluence):
    return 1 + 0.5 * (1 - np.exp(-0.05 * er_fluence))  # saturates at 1.5x boost

Z = np.zeros_like(ER)
for i in range(len(mito_fluence)):
    for j in range(len(er_fluence)):
        m = mito_fluence[i]
        e = er_fluence[j]
        mito_effect = mito_response(m * crosstalk_gain(e))
        er_effect = er_response(e)
        Z[i, j] = np.clip(mito_effect + er_effect, 0, 1)

# 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(ER, MITO, Z, cmap='viridis')
ax.set_xlabel('ER fluence')
ax.set_ylabel('mitochondrial fluence')
ax.set_zlabel('X measure of cell death (more is more dead)')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
for m_level in [10, 30, 50, 70, 90]:
    m_effect = mito_response(m_level * crosstalk_gain(er_fluence))
    e_effect = er_response(er_fluence)
    death = np.clip(m_effect + e_effect, 0, 1)
    ax.plot(er_fluence, death, label=f'Mito fluence = {m_level:.0f} mJ/µm²')
ax.set_xlabel('ER Fluence (mJ/µm²)')
ax.set_ylabel('Apoptosis Probability')
ax.set_title('ER Contribution at Varying Mitochondrial Fluence Levels')
ax.legend()
plt.tight_layout()
plt.show()
