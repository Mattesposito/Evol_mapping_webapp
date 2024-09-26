import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon=":handshake:"
)

st.title('Welcome!')
st.title('\n')

st.markdown('This is a Streamlit app that lets you visualize how the linear matter power spectrum computed with CAMB (https://github.com/cmbant/CAMB) depends on cosmological parameters.')
st.markdown('It consists of two pages: :red[Play with linear Pk] and :red[Create linear Pk GIF]. (You can find them in the sidebar.)')
st.markdown('The first page allows you to play with the parameters of the linear matter power spectrum and visualize the results.')
st.markdown('The second page allows you to create a (downloadable) GIF of the linear matter power spectrum varying one parameter at a time.')

st.markdown('For any request, for reporting bugs or if you would like to take a look at the source code, you can contact me at: [esposito@mpe.mpg.de](mailto:esposito@mpe.mpg.de)')
st.markdown('Enjoy! :smile:')