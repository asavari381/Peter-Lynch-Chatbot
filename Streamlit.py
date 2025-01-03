import streamlit as st

st.image("background.png", use_container_width=True)

st.markdown(
    """
    <h1 style="color: darkgreen; text-align: center; font-size: 45px; font-weight: bold;">
        I am Peter Lynch! Ask me anything
    </h1>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_input("Your question:")
if user_input:
    st.markdown(f"**You asked:** {user_input}")