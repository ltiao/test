
import streamlit as st
import pandas as pd
import altair as alt

import requests
import zipfile
import io

from streamlit.logger import get_logger


logger = get_logger(__name__)

# url = "https://cdn-charts.streeteasy.com/Master%20Report.zip"


@st.cache_data
def get_data():

    r = requests.get("https://cdn-charts.streeteasy.com/rentals/All/medianAskingRent_All.zip")
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content), 'r') as archive:
        logger.info(f"{archive}")
        logger.info(f"Archive contents: {archive.namelist()}")
        with archive.open('medianAskingRent_All.csv', 'r') as infile:
            return pd.read_csv(infile)


frame = get_data().query("areaType == 'neighborhood'") \
                  .melt(id_vars=['Borough', 'areaName', 'areaType'], 
                        var_name='month', value_name='price') \
                  .assign(date=lambda row: pd.to_datetime(row['month'], format='%Y-%m'))

selection = alt.selection_point(fields=['Borough'], bind='legend')

chart = alt.Chart(
    frame,
    title=alt.Title(
       "New York City Real Estate Market",
       subtitle=[
           "Historical monthly data by borough and neighborhood",
           "from StreetEasy"
       ],
       anchor='start',
       orient='top',
       offset=20)
    ) \
    .mark_line(interpolate="step-after", clip=True) \
    .encode(
        alt.X('yearmonth(date):T'), 
        alt.Y('price:Q', axis=alt.Axis(format="$,.0f"))
        .scale(zero=False, domain=(None, 10_000))
        .title('median asking rent'),
        alt.Color('Borough:N'),
        alt.Detail('areaName:N')
        .title('Neighborhood'),
        opacity=alt.condition(selection, alt.value(1.), alt.value(.1))
        # .legend(None)
        # .scale(scheme='tableau10')
    ).add_params(
        selection
    )

st.altair_chart(chart, use_container_width=True)

with st.expander("Raw data", expanded=False):
    st.dataframe(frame, hide_index=True, use_container_width=True)
