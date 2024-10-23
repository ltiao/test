
import streamlit as st

import pandas as pd
import numpy as np

import torch
import tensorflow as tf
import tensorflow_probability as tfp

import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

from botorch.models.kernels import ExponentialDecayKernel
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


torch.manual_seed(SEED)
random_state = np.random.RandomState(SEED)


logger = get_logger(__name__)


def default_float():
    return torch.double


st.set_page_config(page_title="Power Law Decay Kernel", page_icon="🌍")

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

    with st.expander("Advanced options", expanded=True):

        inverse_length_scale = torch.tensor(
            .1 ** st.number_input("$\\log_{10}$ lengthscale", value=0., step=.1),
            dtype=default_float()
        )

        power = torch.tensor(
            10. ** st.number_input("$\\log_{10}$ power", value=0.5, step=.1),
            dtype=default_float()
        )
        offset = torch.tensor(
            st.number_input("offset", value=0.1, step=.1, min_value=1e-9),
            dtype=default_float()
        )

    with st.expander("Plotting options"):
        n_rows = st.number_input("number of rows", min_value=1, value=N_ROWS_VALUE, step=1)
        n_test = int(1 << st.number_input("$\\log_2$ number of test points", min_value=0, value=LOG2_N_TEST_VALUE, step=1))
        n_samples = st.number_input("number of samples", min_value=1, max_value=10, value=N_SAMPLES_VALUE, step=1)

    st.markdown("A dashboard :bar_chart: built :hammer_and_wrench: by [Louis Tiao](https://tiao.io/) " 
                "([Twitter/X @louistiao](https://x.com/louistiao), " 
                "[GitHub @ltiao](https://github.com/ltiao))")

x1 = torch.linspace(0., 10., n_test)
x2 = torch.linspace(0., 10., n_rows)

X1 = x1[..., np.newaxis]
X2 = x2[..., np.newaxis]

kernel_cls = ExponentialDecayKernel
kernel = kernel_cls()
kernel.lengthscale = 1./inverse_length_scale
kernel.power = power
kernel.offset = offset

logger.info(f"kernel: {kernel}")


K = kernel(X1, X2).numpy()

frames = [pd.DataFrame(dict(x1=x1.numpy(), x2=x.numpy(), k=K[..., i])) 
          for i, x in enumerate(x2)]

data = pd.concat(frames, axis="index", sort=True)

logger.info(f"data: {data}")


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

K = kernel(X1, X1)
L = K.cholesky()

eps = torch.randn(n_samples, n_test)

logger.info(f"Normal sample: {eps}")

Y = L.matmul(eps.unsqueeze(-1)).squeeze(-1).detach().numpy()

frames = [pd.DataFrame(dict(sample=i+1, x=x1.numpy(), y=Y[i])) 
          for i in range(n_samples)]
data = pd.concat(frames, axis="index", sort=True)

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

st.altair_chart(samples_chart & kernel_ridgeline_chart)
# st.altair_chart(kernel_ridgeline_chart)
