
import streamlit as st

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from streamlit.logger import get_logger
from matplotlib.ticker import PercentFormatter

logger = get_logger(__name__)


WIDTH = 3.6
ASPECT = 1.6
HEIGHT = WIDTH / ASPECT
DPI = 300
TRANSPARENT = False 
USE_TEX = True
CONTEXT = "paper"
STYLE = "white"
PALETTE = "crest"


def setup_aesthetic(width, height, aspect, dpi, transparent, use_tex, context,
                    style, palette):

    figsize = width, height

    rc = {
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "font.serif": ["Palatino", "Times", "Computer Modern"],
        "text.usetex": use_tex,
        "text.latex.preamble": r"\usepackage{bm}" 
                               r"\usepackage{nicefrac}",
        "savefig.dpi": dpi,
        "savefig.transparent": transparent,
        "axes.facecolor": (0, 0, 0, 0)
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)
    return figsize


width, height = setup_aesthetic(WIDTH, HEIGHT, ASPECT, DPI, TRANSPARENT, 
                                USE_TEX, CONTEXT, STYLE, PALETTE)


st.set_page_config(page_title="Marginal Income Tax", page_icon="🌍")

st.sidebar.header("Marginal Income Tax")


st.markdown("# Marginal Income Tax")

tax_schedule = dict(
    bracket_start_step=[0., 11_000., 33_725., 50_650., 86_725., 49_150., 346_875.],
    rate=[.10, .12, .22, .24, .32, .35, .37]
)

tax_schedule_frame = pd.DataFrame(tax_schedule)
tax_schedule_frame = st.data_editor(
    tax_schedule_frame, 
    column_config=dict(
        bracket_start_step=st.column_config.NumberColumn(
            "Bracket Start Increment",
            format="$%.2f",
            required=True,
            min_value=0.,
        ),
        rate=st.column_config.NumberColumn(
            "Marginal Rate",
            required=True,
            min_value=0.,
            max_value=1.,
            format="%.4f"
        )
    ),
    num_rows='dynamic'
)

tax_schedule_frame = tax_schedule_frame.assign(
    bracket_start=lambda row: row.bracket_start_step.cumsum(),
    bracket_base=lambda row: (row.bracket_start_step * 
                              row.rate.shift(+1, fill_value=0.)).cumsum()
)

st.write(tax_schedule_frame)

# income = np.logspace(3., 8., num=64)  # adjusted gross income
income = np.arange(500., 10_000_000., 2_500)  # adjusted gross income

# nonnegative amount by which income exceeds the start of each bracket
income_over_bracket_start = np.maximum(0., np.subtract.outer(income, tax_schedule_frame.bracket_start.to_numpy()))
income_by_bracket = np.minimum(income_over_bracket_start, tax_schedule_frame.bracket_start_step.shift(-1, fill_value=np.inf))

np.testing.assert_array_equal(income, income_by_bracket.sum(axis=-1))

st.write(income_by_bracket * tax_schedule_frame.rate.to_numpy())



# .clip(
#     tax_schedule_frame.bracket_start, 
#     tax_schedule_frame.bracket_start.shift(-1, fill_value=np.inf)
# )


# st.write(income_over_bracket_start.clip(0., tax_schedule_frame.bracket_start_step.shift(-1, fill_value=np.inf)))

# st.write(np.minimum(np.maximum(0., income[..., np.newaxis] - tax_schedule_frame.bracket_start.to_numpy()), tax_schedule_frame.bracket_start_step.shift(-1, fill_value=np.inf)))
# st.write((income[..., np.newaxis] - tax_schedule_frame.bracket_start.to_numpy()).clip(
#         0.,
#         tax_schedule_frame.bracket_start_step.shift(-1, fill_value=np.inf)
#     )
# )

st.write(tax_schedule_frame.bracket_start_step.shift(-1, fill_value=np.inf))
st.write(tax_schedule_frame.bracket_start.shift(-1) - tax_schedule_frame.bracket_start)

logger.info(f"income: {income.shape}")

logger.info(f"tax_schedule_frame.bracket_start: {tax_schedule_frame.bracket_start}")
logger.info(f"tax_schedule_frame.bracket_start: {tax_schedule_frame.bracket_start.shift(-1, fill_value=None).to_numpy()}")

# st.write(income_frame.clip(tax_schedule_frame.bracket_start, 
#                            tax_schedule_frame.bracket_start.shift(+1), 
#                            axis='columns'))

# st.write(income[..., np.newaxis] - tax_schedule_frame.bracket_start.to_numpy())
# st.write(np.minimum(tax_schedule_frame.bracket_start.shift(-1).to_numpy(), np.maximum(tax_schedule_frame.bracket_start.to_numpy(), income[..., np.newaxis])) - tax_schedule_frame.bracket_start.to_numpy())

# bracket_index = tax_schedule_frame.bracket_start.searchsorted(income, side='left') - 1

# tax_liability = tax_schedule_frame.bracket_base.iloc[bracket_index] + \
#     tax_schedule_frame.rate.iloc[bracket_index] * \
#     (income - tax_schedule_frame.bracket_start.iloc[bracket_index])

# data = pd.DataFrame(dict(income=income, tax_liability=tax_liability)) \
#     .assign(effective_rate=lambda row: row.tax_liability / row.income,
#             net_income=lambda row: row.income - row.tax_liability)

# st.write(data)

# fig, ax = plt.subplots()

# sns.lineplot(x='income', y="effective_rate",
#              # hue=r"$\omega$", 
#              # style='kind',
#              # hue=r'$L$', hue_norm=LogNorm(),
#              # units='component', estimator=None,
#              # legend='brief',
#              # linewidth=.1, alpha=.8,
#              # palette='vlag',
#              data=data, ax=ax)

# ax.set_xscale('log')
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.))

# ax.set_xlabel('taxable income (\\$)')
# ax.set_ylabel('effective tax rate')

# st.pyplot(fig=fig, clear_figure=True, use_container_width=True)

st.divider()

st.text(
    """
    DISCLAIMER: This tax calculator provides an estimate based on the 
    information you input and current tax laws. It is intended for 
    informational purposes only and should not be considered tax, legal, or 
    financial advice.

    The results provided by this calculator are not a substitute for 
    professional advice from a qualified tax professional, attorney, or 
    financial advisor. The accuracy of the calculations depends on the accuracy
    of the information you provide and the assumptions made by the calculator.

    Tax laws and regulations are complex and subject to change. We make no 
    guarantees regarding the accuracy, completeness, or reliability of the 
    information provided by this calculator. We assume no liability for any 
    errors, omissions, or inconsistencies in the results.

    Your use of this calculator is at your own risk. We are not responsible for
    any damages, losses, or expenses incurred as a result of using this 
    calculator or relying on its results. It is your responsibility to verify 
    the accuracy of the results and ensure compliance with applicable tax laws 
    and regulations.
    
    For accurate calculations and personalized advice specific to your unique 
    tax situation, please consult a qualified tax professional or the official 
    resources provided by the relevant tax authorities.
    
    By using this tax calculator, you acknowledge that you have read, 
    understood, and agree to be bound by this disclaimer.
    """
)
