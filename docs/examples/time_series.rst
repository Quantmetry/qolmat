############################################
Example for anomaly detection in time series
############################################

The aim of this notebook is to make use of the RPCA method to
detect anoamlies in an univariate time series. 
Note that this method only applies whether the time series has 
seasonalities or periodicities/structures in itself.

First, import some usefull libraries and functions

.. code-block:: python

    import numpy as np
    import timesynth as ts # package for generating time series

    import matplotlib.pyplot as plt

    from robust_pca.utils import drawing, utils
    from robust_pca.classes.pcp_rpca import PcpRPCA
    from robust_pca.classes.temporal_rpca import TemporalRPCA, OnlineTemporalRPCA


Then we generate some synthetic data. More precisely, we consider a sine function 
which we corrupt by adding some anomalies and creating some missing values.

.. code-block:: python

    time_sampler = ts.TimeSampler(stop_time=20)
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=5_000, keep_percentage=100)
    sinusoid = ts.signals.Sinusoidal(frequency=2)
    white_noise = ts.noise.GaussianNoise(std=0.1)
    timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(irregular_time_samples)

    n = len(samples)
    pc = 0.01
    indices = np.random.choice(n, int(n*pc))
    samples[indices] = [np.random.uniform(low=2*np.min(samples), high=2*np.max(samples)) for i in range(int(n*pc))]
    indices = np.random.choice(n, int(n*pc))
    samples[indices] = np.nan

    time_sampler = ts.TimeSampler(stop_time=20)
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=5_000, keep_percentage=100)
    sinusoid = ts.signals.Sinusoidal(frequency=3)
    white_noise = ts.noise.GaussianNoise(std=0.1)
    timeseries = ts.TimeSeries(sinusoid)#, noise_generator=white_noise)
    samples2, signals2, errors2 = timeseries.sample(irregular_time_samples)

    n2 = len(samples2)
    indices = np.random.choice(n2, int(n*pc))
    samples2[indices] = [np.random.uniform(low=2*np.min(samples2), high=2*np.max(samples2)) for i in range(int(n2*pc))]
    indices = np.random.choice(n2, int(n*pc))
    samples2[indices] = np.nan

    samples += samples2
    signals += signals2
    errors += errors2

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12,6))
    ax[0].plot(range(n), samples)
    ax[0].set_title("Corrupted signal", fontsize=15)
    ax[1].plot(range(n), signals)
    ax[1].set_title("Low-rank signal", fontsize=15)
    ax[2].plot(range(n), errors)
    ax[2].set_title("Noise", fontsize=15)
    ax[2].set_xlabel("Time", fontsize=16)
    plt.tight_layout()
    plt.show()

.. image:: ../images/time_series_01.png

The aim is to find the smooth signal (Low-rank signal) as well as the anomalies given 
the observed signal (Corrupted signal).

We first try the basic RPCA formulation, where we do not take into account the temporal aspect of data under scrutiny.
We then apply the batch version of temporal RPCA.
Then, suppose we only have access to some data but new samples arrive constantly. 
We do not want to compute a RPCA from scratch each time new data come, but we want 
to use the knowledge we have from the precedent one. This is a perfect scenario 
for using the online version of the algorithm. In this example, we take as burning 
sample the 40% first percent of the time series. 
For the online version, we test with and without a moving window.


.. code-block:: python

    a = PcpRPCA(period=25)
    a.fit(signal=samples.tolist())
    Afilter, noise = utils.get_anomaly(a.A, a.X, e=2)
    s1_pcp, s2_pcp, s3_pcp = utils.resultRPCA_to_signal(a.D, a.X, Afilter, a.rest)

    a = TemporalRPCA(period=25, lam1=2, lam2=0.3, list_periods=[20], list_etas=[0.01], norm="L2")
    a.fit(signal=samples.tolist())
    s1_temp, s2_temp, s3_temp = utils.resultRPCA_to_signal(a.D, a.X, a.A, a.rest)

    a = OnlineTemporalRPCA(period=25, lam1=2, lam2=0.4, list_periods=[20], list_etas=[0.01], norm="L2",
                        burnin=0.4, online_list_periods=[20], online_list_etas=[0.2])
    a.fit(signal=samples.tolist())
    s1_on, s2_on, s3_on = utils.resultRPCA_to_signal(a.D, a.X, a.A, a.rest)

    a = OnlineTemporalRPCA(period=25, lam1=2, lam2=0.4, list_periods=[20], list_etas=[0.01], norm="L2",
                        burnin=0.4, nwin=50, online_list_periods=[20], online_list_etas=[0.2])
    a.fit(signal=samples.tolist())
    s1_onw, s2_onw, s3_onw = utils.resultRPCA_to_signal(a.D, a.X, a.A, a.rest)

Let's take a look at these results.

.. code-block:: python

    fs = 15
    colors = ["darkblue", "tab:red"]

    fig, ax = plt.subplots(4, 2, sharex=True,  sharey=False, figsize=(20,8))
    for j, s in enumerate(zip([s2_pcp, s3_pcp], [s2_temp, s3v], [s2_on, s_on], [s2_onw, s_onw])):
        for i,e in enumerate(s):
            ax[i][j].plot(x, e, c=colors[j])
            ax[i][j].set_yticks([-2, 0, 2])
            ax[i][j].tick_params('both', length=8, width=1, which='major')
        
    for i,y in enumerate(["PCP", "Temporal\n batch", "Temporal\n Online", "Temporal\n Online\n Moving Window"]):
        ax[i][0].set_ylabel(f"{y} \n\ny", fontsize=fs)
        ax[i][1].set_ylabel("outliers ampl.", fontsize=fs)
    ax[3][0].set_xlabel("Time", fontsize=fs)
    ax[3][1].set_xlabel("Time", fontsize=fs)

    plt.tight_layout()
    plt.show()

.. image:: ../images/time_series_05.png

One sees the reconstruction for the online part is a little bit more noisy. 
However, the anomalies are well detected, and it is much more faster!


.. note::
    Since in the problem formulation, the data fitting is no more a constraint, 
    the sparse part is immediately sparser than in classic formulation. 
    We do not need a filering step to extract the biggest anoamlies (in amplitude).
    However, we do not have anymore the equality :math:`D = X + A`. 

.. warning::
    The quality of signal reconstruction and anomaly detection 
    just as the transition from batch to online processing 
    is greatly improvable.

