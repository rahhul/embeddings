import streamlit as st
import pandas as pd
import numpy as np

st.title('My hello app')

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))




map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [20, 20] + [-33.86, 151.20],
    columns=['lat', 'lon'])


st.write(map_data)
st.map(map_data)
