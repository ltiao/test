
import streamlit as st
import numpy as np
import pandas as pd

from streamlit.logger import get_logger

logger = get_logger(__name__)


SEED_VALUE = 8888
MIN_PLAYERS = 3
WONDERS_OPTIONS = [
    "Alexandria",
    "Babylon",
    "Ephesos",
    "Giza",
    "Halicarnassus",
    "Olympia",
    "Rhodes",
    # Leaders
    "Abu Simbel",
    "Roma",
    # Cities
    "Petra",
    "Byzantium",
    # Edifice
    "Ur",
    "Carthage",
    # Armada
    "Siracusa"
]
WONDERS_EXCLUDES_DEFAULT = ["Halicarnassus", "Olympia"]

st.set_page_config(page_title="7 Wonders", page_icon="🌍")

st.title("7 Wonders Configuration Generator")
st.image("https://cdn.svc.asmodee.net/production-rprod/storage/downloads/games/7wonders/wallpapers/desktop/7wonders-wallpaper-1-1598892294KBXAc.jpg",
         use_column_width=True)


with st.form("my_form"):

    st.caption("Settings")

    frame = pd.DataFrame(dict(name=['Foo', 'Bar', 'Baz']))
    frame = st.data_editor(
        frame, 
        column_config=dict(
            name=st.column_config.TextColumn(
                "Player Name 👤",
                help="Name of the player?",
            )
        ),
        num_rows="dynamic",
        use_container_width=True
    )

    seed = st.number_input("Random Seed", min_value=0, max_value=(1 << 32) - 1,
                           value=SEED_VALUE, step=1,
                           help="Number to seed random number generator.")
    random_state = np.random.RandomState(seed)

    wonders_excludes = st.multiselect("Exclude Wonders", WONDERS_OPTIONS, 
                                      default=WONDERS_EXCLUDES_DEFAULT)
    wonders = set(WONDERS_OPTIONS) - set(wonders_excludes)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Generate")

if submitted:

    n_players = len(frame)
    n_wonders = len(wonders)

    if n_players < MIN_PLAYERS:
        st.error(f"Must have {MIN_PLAYERS:d} or more players ({n_players:d})!")
        st.stop()

    if n_wonders < n_players:
        st.error(f"Number of players ({n_players:d}) exceeds number of "
                 f"Wonders ({n_wonders:d})!")
        st.stop()

    wonder_choice = random_state.choice(np.fromiter(wonders, dtype="<U16"), 
                                        size=n_players, replace=False)

    with st.container(border=True):
        st.caption("Configuration")
        st.dataframe(frame.assign(wonder=wonder_choice)
                          .sample(frac=1., ignore_index=True,
                                  random_state=random_state)
                          .reset_index(), 
                     column_config=dict(
                        index=st.column_config.NumberColumn("Seat 🪑"),
                        name=st.column_config.TextColumn("Player Name 👤"),
                        wonder=st.column_config.TextColumn("Wonder Board 🗿")
                     ),
                     hide_index=True,
                     use_container_width=True)
