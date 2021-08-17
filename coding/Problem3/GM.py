from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from coding.Problem3.draw_utils import plot_gaussian_mixture
from coding.utils import getLocation

plt.figure()
X_DataFrame = getLocation()
X_DataFrame
X_train = X_DataFrame.values
gm = GaussianMixture(n_components=8,random_state=42)
gm.fit(X_train)






plt.figure(figsize=(8, 4))

plot_gaussian_mixture(gm, X_train)

# save_fig("gaussian_mixtures_plot")
plt.show()