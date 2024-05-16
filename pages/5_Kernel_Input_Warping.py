
import streamlit as st

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial
from streamlit.logger import get_logger

rc = {
    # "figure.figsize": figsize,
    # "figure.dpi": dpi,
    # "font.serif": ["Palatino", "Times", "Computer Modern"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{bm}" 
                           r"\usepackage{nicefrac}",
    # "savefig.dpi": dpi,
    "savefig.transparent": True,
    # "axes.facecolor": (0, 0, 0, 0)
}
sns.set_theme(  # context=context, style=style, palette=palette, 
    style="white",
    font="serif", 
    rc=rc
)

N_TEST = 1 << 9
N_ROWS = 8

OBSERVATION_NOISE_VARIANCE = 0.
JITTER = 1e-9
SEED = 42


random_state = np.random.RandomState(SEED)

logger = get_logger(__name__)

tfd = tfp.distributions
kernels = tfp.math.psd_kernels


kernel_classes = dict(
    sqrexp=kernels.ExponentiatedQuadratic,
    matern12=kernels.MaternOneHalf,
    matern32=kernels.MaternThreeHalves,
    matern52=kernels.MaternFiveHalves,
)

kernel_names = dict(
    sqrexp="Squared Exponential",
    matern12="Matern-1/2",
    matern32="Matern-3/2",
    matern52="Matern-5/2",
)


def default_float():
    return "float64"


st.set_page_config(page_title="Input Warping", page_icon="🌍")

# st.sidebar.header("Kernel ")

st.markdown(
    """
    Input warping is an augmentation technique used in Gaussian processes (GPs) 
    to model nonstationary data by transforming the input space such that 
    regions of rapid change can be effectively captured. 
    This is particularly useful when the underlying function exhibits varying 
    degrees of smoothness or different behaviors across its domain. 

    The core idea is to apply a monotonic transformation to the inputs using 
    a cumulative distribution function (CDF), which can warp the inputs 
    non-linearly and thereby adapt the GP to the nonstationarities in the data.

    A common choice for the transformation is the CDF of the Beta or 
    Kumaraswamy distribution, both of which are flexible in shaping the warping 
    through their parameters. 

    For instance, using the Kumaraswamy CDF 

    $$
    w(x; a, b) = 1 - (1-x^a)^b
    $$ 

    where $a$ and $b$ are shape parameters, allows for different degrees of 
    input compression or expansion. 

    This transformation alters the effective "length-scale" locally, enabling 
    the GP to vary its sensitivity to changes in different regions of the 
    input space. 
    By using such transformations, GPs can better model functions that are 
    smooth in some areas but erratic in others, overcoming a traditional 
    limitation in handling nonstationary behaviors within a unified modeling 
    framework.
    """
)

x1 = np.linspace(0., 1., N_TEST)
x2 = np.linspace(0., 1., N_ROWS)

X1 = x1[..., np.newaxis]
X2 = x2[..., np.newaxis]

with st.sidebar:
    kernel_key = st.radio("Base kernel $k(x, x')$", 
                          options=kernel_names.keys(),
                          format_func=kernel_names.get)

    log10_length_scale_value = tf.convert_to_tensor(
        st.number_input("lengthscale ($\\log_{10}{\\ell}$)", value=-1., step=.1),
        dtype=default_float()
    )
    log10_amplitude_value = tf.convert_to_tensor(
        st.number_input("amplitude ($\\log_{10}{\\alpha}$)", value=0., step=.1),
        dtype=default_float()
    )

    st.divider()
    st.header("Warping Function")
    st.subheader("Kumaraswamy CDF")

    ln_concentration1_value = tf.convert_to_tensor(
        st.number_input("shape ($\\ln{a}$)", value=0., step=.25),
        dtype=default_float()
    )
    ln_concentration0_value = tf.convert_to_tensor(
        st.number_input("shape ($\\ln{b}$)", value=0., step=.25),
        dtype=default_float()
    )

    st.latex(
        f""" 
        w(x) = 1 - (1-x^{{{np.exp(ln_concentration1_value):.3f}}})^{{{np.exp(ln_concentration0_value):.3f}}}
        """
    )

    kernel_class = kernel_classes[kernel_key]
    kernel = kernel_class(
        inverse_length_scale=.1**log10_length_scale_value,
        amplitude=10.**log10_amplitude_value,
    )

    kernel_warped = tfp.math.psd_kernels.KumaraswamyTransformed(
        kernel,
        concentration1=tf.exp(ln_concentration1_value),
        concentration0=tf.exp(ln_concentration0_value),
    )

    logger.info(f"Kernel (base): {kernel_warped._kernel}")
    logger.info(f"Kernel (input warped): {kernel_warped}")
    logger.info(f"Warping function: {kernel_warped._transformation_fn}")

    warping_fn = partial(kernel_warped._transformation_fn,
                         feature_ndims=1, example_ndims=1)

    logger.info(f"warping_fn(x1): {warping_fn(x1).shape}")

    # bijector = tfp.bijectors.KumaraswamyCDF(
    #     concentration1=tf.exp(ln_concentration1_value),
    #     concentration0=tf.exp(ln_concentration0_value),
    # )

    data = pd.DataFrame({'x': x1, 'w(x)': warping_fn(x1)})

    warping_chart = alt.Chart(data) \
        .mark_line() \
        .encode(
            alt.X('x:Q'), 
            alt.Y('w(x):Q'),
        ).properties(
            height=200,
            width=200
        )

    # st.altair_chart(chart, use_container_width=True)

    st.divider()

    st.markdown("A demo by [Louis Tiao](https://tiao.io/) (Twitter/X @louistiao, GitHub @ltiao)")

K = kernel_warped.matrix(X1, X2).numpy()

frames = [pd.DataFrame(dict(x1=x1, x2=u, k=K[..., i])) 
          for i, u in enumerate(x2)]

data = pd.concat(frames, axis="index", sort=True)

# st.dataframe(data, use_container_width=True)

# palette = sns.cubehelix_palette(N_ROWS, rot=-.25, light=.7)
# g = sns.FacetGrid(data, row="x2", hue="x2", aspect=8., height=1.2, palette=palette)


# # Define and use a simple function to label the plot in axes coordinates
# def label(x, color, label):
#     ax = plt.gca()
#     ax.text(0., .2, f"$x'={float(label):.2f}$", fontweight="bold", color=color,
#             ha="left", va="center", transform=ax.transAxes)


# def fn(x, k, color, label, *args, **kwargs):
#     ax = plt.gca()
#     ax.fill_between(x, k, color=color, alpha=.95)


# g.map(fn, "x1", "k")
# g.map(plt.plot, "x1", "k", color='w', lw=2.)

# g.refline(y=0., linewidth=2., linestyle="-", color=None, clip_on=False)
# g.map(label, "x1")

# g.figure.subplots_adjust(hspace=-.5)

# g.set_titles("")
# g.set(yticks=[], ylabel="")
# g.despine(bottom=True, left=True)

# st.pyplot(fig=g.figure, clear_figure=True, use_container_width=True)

step = 32.
overlap = 1.

kernel_ridgeline_chart = alt.Chart(data, height=step) \
    .mark_area(interpolate='linear',
               fillOpacity=0.8,
               stroke='white',
               strokeWidth=1.6) \
    .encode(alt.X('x1:Q').title('x'), 
            alt.Y('k:Q')
               .axis(None)
               .scale(range=[+step, -step * overlap]),
            alt.Fill('x2:Q')
               .legend(None)
               .scale(scheme='tealblues')) \
    .facet(row=alt.Row('x2:Q')
                  .title("x'")
                  .header(labelAngle=0., labelAlign='left', format='.3f')) \
    .properties(bounds='flush', 
                # title='Input-warped covariance function',
                )
    # .configure_facet(spacing=0.) \
    # .configure_axis(grid=False) \
    # .configure_view(stroke=None) \
    # .configure_title(anchor='end')

gaussian_process = tfd.GaussianProcess(
    kernel_warped,
    index_points=X1,
    observation_noise_variance=OBSERVATION_NOISE_VARIANCE,
    jitter=JITTER,
)
logger.info(f"Gaussian process: {gaussian_process}")

N_SAMPLES = 3

eps = random_state.randn(N_SAMPLES, N_TEST)

logger.info(f"Normal sample: {eps}")

Y = gaussian_process.get_marginal_distribution() \
    .bijector.forward(eps).numpy()

st.write(Y)

frames = [pd.DataFrame(dict(sample=i+1, x=x1, y=Y[i])) 
          for i in range(N_SAMPLES)]
data = pd.concat(frames, axis="index", sort=True)

st.write(data)

samples_chart = alt.Chart(data) \
    .mark_line(clip=True) \
    .encode(
        alt.X('x:Q'), 
        alt.Y('y:Q')
        .title('f(x)')
        .scale(domain=(-3.6, +3.6)),
        alt.Color('sample:N')
        .legend(None)
        # .scale(scheme='tableau10')
    ).properties(
        height=300,
        width=600
    )

st.altair_chart(samples_chart & (warping_chart | kernel_ridgeline_chart))
