"""Cross-page navigation for the investigation workflow
(transaction → account → network → ring → action)."""

import streamlit as st

PAGES = {}


def go(page, **selection):
    for k, v in selection.items():
        st.session_state[k] = v
    st.switch_page(PAGES[page])


def button(label, page, key, **selection):
    if st.button(label, key=key, width="stretch"):
        go(page, **selection)


def pick(options, state_key, label, fmt=None, key=None):
    """Selectbox that defaults to the entity chosen elsewhere in the app."""
    options = list(options)
    if not options:
        return None
    current = st.session_state.get(state_key)
    idx = options.index(current) if current in options else 0
    value = st.selectbox(label, options, index=idx, format_func=fmt or str, key=key)
    st.session_state[state_key] = value
    return value
