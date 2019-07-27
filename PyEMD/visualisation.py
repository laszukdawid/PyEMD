import numpy as np

# Visualisation is an optional module. To minimise installation, `matplotlib` is not added
# by default. Please install extras with `pip install -r requirement-extra.txt`.
try:
    import pylab as plt
    from scipy.signal import hilbert
except ImportError:
    pass


class Visualisation(object):
    """Simple visualisation helper.

    This class is for quick and simple result visualisation.
    """

    PLOT_WIDTH = 6
    PLOT_HEIGHT_PER_IMF = 1.5

    def __init__(self, emd_instance=None):
        self.emd_instance = emd_instance

        self.imfs = None
        self.residue = None

        if emd_instance is not None:
            self.imfs, self.residue = self.emd_instance.get_imfs_and_residue()

    def _check_imfs(self, imfs, residue, include_residue):
        """Checks for passed imfs and residue."""
        imfs = imfs if imfs is not None else self.imfs
        residue = residue if residue is not None else self.residue

        if imfs is None:
            raise AttributeError("No imfs passed to plot")

        if include_residue and residue is None:
            raise AttributeError("Requested to plot residue but no residue provided")

        return imfs, residue

    def plot_imfs(self, imfs=None, residue=None, t=None, include_residue=True):
        """Plots and shows all IMFs.

        All parameters are optional since the `emd` object could have been passed when instantiating this object.

        The residual is an optional and can be excluded by setting `include_residue=False`.
        """
        imfs, residue = self._check_imfs(imfs, residue, include_residue)

        num_rows, t_length = imfs.shape
        num_rows += include_residue is True

        t = t if t is not None else range(t_length)

        fig, axes = plt.subplots(num_rows, 1, figsize=(self.PLOT_WIDTH, num_rows*self.PLOT_HEIGHT_PER_IMF))

        if num_rows == 1:
            axes = list(axes)

        axes[0].set_title("Time series")

        for num, imf in enumerate(imfs):
            ax = axes[num]
            ax.plot(t, imf)
            ax.set_ylabel("IMF " + str(num+1))

        if include_residue:
            ax = axes[-1]
            ax.plot(t, residue)
            ax.set_ylabel("Res")

        # Making the layout a bit more pleasant to the eye
        plt.tight_layout()

    def plot_instant_freq(self, t, imfs=None):
        """Plots and shows instantaneous frequencies for all provided imfs.

        The necessary parameter is `t` which is the time array used to compute the EMD.
        One should pass `imfs` if no `emd` instances is passed when creating the Visualisation object.
        """
        imfs, _ = self._check_imfs(imfs, None, False)
        num_rows = imfs.shape[0]

        imfs_inst_freqs = self._calc_inst_freq(imfs, t)

        fig, axes = plt.subplots(num_rows, 1, figsize=(self.PLOT_WIDTH, num_rows*self.PLOT_HEIGHT_PER_IMF))

        if num_rows == 1:
            axes = list(axes)

        axes[0].set_title("Instantaneous frequency")

        for num, imf_inst_freq in enumerate(imfs_inst_freqs):
            ax = axes[num]
            ax.plot(t[:-1], imf_inst_freq)
            ax.set_ylabel("IMF {} [Hz]".format(num+1))

        # Making the layout a bit more pleasant to the eye
        plt.tight_layout()

    def _calc_inst_phase(self, sig):
        """Extract analytical signal through the Hilbert Transform."""
        analytic_signal = hilbert(sig)  # Apply Hilbert transform to each row
        phase = np.unwrap(np.angle(analytic_signal))  # Compute angle between img and real
        return phase

    def _calc_inst_freq(self, sig, t):
        """Extracts instantaneous frequency through the Hilbert Transform."""
        inst_phase = self._calc_inst_phase(sig)
        return np.diff(inst_phase)/(2*np.pi*(t[1]-t[0]))

    def show(self):
        plt.show()


if __name__ == "__main__":
    import numpy as np
    from EMD import EMD

    # Simple signal example
    t = np.arange(0, 3, 0.01)
    S = np.sin(13*t + 0.2*t**1.4) - np.cos(3*t)

    emd = EMD()
    emd.emd(S)
    imfs, res = emd.get_imfs_and_residue()

    # Initiate visualisation with emd instance
    vis = Visualisation(emd)

    # Create a plot with all IMFs and residue
    vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)

    # Create a plot with instantaneous frequency of all IMFs
    vis.plot_instant_freq(t, imfs=imfs)

    # Show both plots
    vis.show()
