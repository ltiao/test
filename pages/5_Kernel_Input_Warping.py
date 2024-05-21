
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
    # "savefig.dpi": dpi,
    "savefig.transparent": True,
    # "axes.facecolor": (0, 0, 0, 0)
}
sns.set_theme(  # context=context, style=style, palette=palette, 
    style="white",
    font="serif", 
    rc=rc
)

LOG2_N_TEST_VALUE = 9
N_ROWS_VALUE = 6
N_SAMPLES_VALUE = 5

OBSERVATION_NOISE_VARIANCE = 0.
JITTER = 1e-9
SEED = 42


random_state = np.random.RandomState(SEED)

logger = get_logger(__name__)

tfd = tfp.distributions
kernels = tfp.math.psd_kernels


kernel_classes = dict(
    matern12=kernels.MaternOneHalf,
    matern32=kernels.MaternThreeHalves,
    matern52=kernels.MaternFiveHalves,
    sqrexp=kernels.ExponentiatedQuadratic,
)

kernel_names = dict(
    matern12="Matern-1/2",
    matern32="Matern-3/2",
    matern52="Matern-5/2",
    sqrexp="Squared Exponential",
)


def default_float():
    return "float64"


st.set_page_config(page_title="Input Warping", page_icon="🌍")

# st.sidebar.header("Kernel ")
# st.markdown(
#     """
#     Input warping is an augmentation technique used in Gaussian processes (GPs) 
#     to model nonstationary data by transforming the input space such that 
#     regions of rapid change can be effectively captured. 
#     This is particularly useful when the underlying function exhibits varying 
#     degrees of smoothness or different behaviors across its domain. 

#     The core idea is to apply a monotonic transformation to the inputs using 
#     a cumulative distribution function (CDF), which can warp the inputs 
#     non-linearly and thereby adapt the GP to the nonstationarities in the data.

#     A common choice for the transformation is the CDF of the Beta or 
#     Kumaraswamy distribution, both of which are flexible in shaping the warping 
#     through their parameters. 

#     For instance, using the Kumaraswamy CDF 

#     $$
#     w(x; a, b) = 1 - (1-x^a)^b
#     $$ 

#     where $a$ and $b$ are shape parameters, allows for different degrees of 
#     input compression or expansion. 

#     This transformation alters the effective "length-scale" locally, enabling 
#     the GP to vary its sensitivity to changes in different regions of the 
#     input space. 
#     By using such transformations, GPs can better model functions that are 
#     smooth in some areas but erratic in others, overcoming a traditional 
#     limitation in handling nonstationary behaviors within a unified modeling 
#     framework.
#     """
# )

with st.sidebar:

    with st.expander("Base kernel", expanded=False):

        kernel_key = st.radio("kernel $k(x, x')$", 
                            options=kernel_names.keys(),
                            index=len(kernel_names)-1,
                            format_func=kernel_names.get)

        inverse_length_scale = tf.convert_to_tensor(
            .1 ** st.number_input("$\\log_{10}$ lengthscale", value=-1.2, step=.1),
            dtype=default_float()
        )
        amplitude = tf.convert_to_tensor(
            10. ** st.number_input("$\\log_{10}$ amplitude", value=0., step=.1),
            dtype=default_float()
        )

    with st.expander("Warping function", expanded=True):

    # st.header("")
    # st.subheader("Kumaraswamy CDF")

        ln_shape_1 = tf.convert_to_tensor(
            st.number_input("$\\ln$ shape $a$", value=1., step=.05),
            dtype=default_float()
        )
        ln_shape_0 = tf.convert_to_tensor(
            st.number_input("$\\ln$ shape $b$", value=1., step=.05),
            dtype=default_float()
        )

        st.latex(
            f""" 
            w(x) = 1 - (1-x^{{{np.exp(ln_shape_1):.3f}}})^{{{np.exp(ln_shape_0):.3f}}}
            """
        )

    with st.expander("Advanced options"):
        n_rows = st.number_input("number of rows", min_value=1, value=N_ROWS_VALUE, step=1)
        n_test = int(1 << st.number_input("$\\log_2$ number of test points", min_value=0, value=LOG2_N_TEST_VALUE, step=1))
        n_samples = st.number_input("number of samples", min_value=1, max_value=10, value=N_SAMPLES_VALUE, step=1)

    st.markdown("A dashboard :bar_chart: built :hammer_and_wrench: by [Louis Tiao](https://tiao.io/) " 
                "([Twitter/X @louistiao](https://x.com/louistiao), " 
                "[GitHub @ltiao](https://github.com/ltiao))")

x1 = np.linspace(0., 1., n_test)
x2 = np.linspace(0., 1., n_rows)

X1 = x1[..., np.newaxis]
X2 = x2[..., np.newaxis]


kernel_class = kernel_classes[kernel_key]
kernel = kernel_class(
    inverse_length_scale=inverse_length_scale, 
    amplitude=amplitude,
)

kernel_warped = tfp.math.psd_kernels.KumaraswamyTransformed(
    kernel,
    concentration1=tf.exp(ln_shape_1),
    concentration0=tf.exp(ln_shape_0),
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

eps = random_state.randn(n_samples, n_test)

logger.info(f"Normal sample: {eps}")

Y = gaussian_process.get_marginal_distribution() \
    .bijector.forward(eps).numpy()

# st.write(Y)

frames = [pd.DataFrame(dict(sample=i+1, x=x1, y=Y[i])) 
          for i in range(n_samples)]
data = pd.concat(frames, axis="index", sort=True)

# st.write(data)

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
