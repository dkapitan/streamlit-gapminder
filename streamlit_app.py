from dataclasses import dataclass

import altair as alt
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass
class Gapminder:
    """Class for storing Gapminder data and plots"""

    url: str = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
    year: int = 1952
    show_data: bool = False
    show_legend: bool = True
    chart_height: int = 500

    def __post_init__(self):
        self.dataset = pd.read_csv(self.url)
        self.df = self.get_data()
        self.title = f"Life expectancy vs. GPD ({self.year}"
        self.xlabel = "GDP per capita (2000 dollars)"
        self.ylabel = "Life expectancy (years)"
        self.xlim = (self.df['gdpPercap'].min()-100,self.df['gdpPercap'].max()+1000)
        self.ylim = (20, 90)

    def get_data(self):
        """Return gapminder data for a given year.

        Countries with gdpPercap lower than 10,000 are discarded.
        """
        df = self.dataset[
            (self.dataset.year == self.year) & (self.dataset.gdpPercap < 10000)
        ].copy()
        df["size"] = np.sqrt(df["pop"] * 2.666051223553066e-05)
        return df

    def altair(self):
        legend = {} if self.show_legend else {"legend": None}
        plot = (
            alt.Chart(self.df)
            .mark_circle()
            .encode(
                alt.X(
                    "gdpPercap:Q",
                    scale=alt.Scale(type="log"),
                    axis=alt.Axis(title=self.xlabel),
                ),
                alt.Y(
                    "lifeExp:Q",
                    scale=alt.Scale(zero=False, domain=self.ylim),
                    axis=alt.Axis(title=self.ylabel),
                ),
                size=alt.Size("pop:Q", scale=alt.Scale(type="log"), legend=None),
                color=alt.Color(
                    "continent", scale=alt.Scale(scheme="category10"), **legend
                ),
                tooltip=["continent", "country", "gdpPercap", "lifeExp"],
            )
            .properties(title="Altair", height=self.chart_height)
            .configure_title(anchor="start")
        )

        return plot.interactive()

    def plotly(self):
        traces = []
        for continent, self.df in self.df.groupby("continent"):
            marker = dict(
                symbol="circle",
                sizemode="area",
                sizeref=0.1,
                size=self.df["size"],
                line=dict(width=2),
            )
            traces.append(
                go.Scatter(
                    x=self.df.gdpPercap,
                    y=self.df.lifeExp,
                    mode="markers",
                    marker=marker,
                    name=continent,
                    text=self.df.country,
                )
            )

        axis_opts = dict(
            gridcolor="rgb(255, 255, 255)", zerolinewidth=1, ticklen=5, gridwidth=2
        )
        layout = go.Layout(
            title="Plotly",
            showlegend=self.show_legend,
            height=self.chart_height,
            xaxis=dict(title=self.xlabel, type="log", **axis_opts),
            yaxis=dict(title=self.ylabel, **axis_opts),
        )

        return go.Figure(data=traces, layout=layout)

    
    def bokeh(self):
        # note bokeh version issue https://discuss.streamlit.io/t/bokeh-2-0-potentially-broken-in-streamlit/2025/8
        plot = figure()
        plot.scatter(x=self.df.gdpPercap, y=self.df.lifeExp, radius=self.df.size)
        return plot

    
    def pyplot(self):
            data = self.df
            title = "Matplotlib"
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.set_xscale("log")
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(self.xlabel, fontsize=10)
            ax.set_ylabel(self.ylabel, fontsize=10)
            ax.set_ylim(self.ylim)
            ax.set_xlim(self.xlim)

            for continent, df in data.groupby('continent'):
                ax.scatter(df.gdpPercap, y=df.lifeExp, s=df['size']*5,
                        edgecolor='black', label=continent)
                
            if self.show_legend:
                ax.legend(loc=4)
            
            return fig


# initiate
gapminder = Gapminder()
st.set_page_config(layout="wide")

# side bar
gapminder.year = st.sidebar.slider(label="year", min_value=1952, max_value=2007, step=5)
gapminder.show_legend = st.sidebar.checkbox("Toggle legend", gapminder.show_legend)
gapminder.df = gapminder.get_data()


# main body
st.title("Gapminder in different ways")
st.markdown(
    """Demo of different interactive plotting libraries using the classic
    [gapminder bubble chart](https://discuss.streamlit.io/t/bokeh-2-0-potentially-broken-in-streamlit/2025/8).
    """
)

with st.expander("Show data"):
    st.dataframe(gapminder.df)

col1, col2 = st.columns([1, 1])

with col1:
    st.altair_chart(gapminder.altair(), True)
    st.plotly_chart(gapminder.plotly(), True)

with col2:
    st.pyplot(gapminder.pyplot(), False)

# st.bokeh_chart(gapminder.bokeh())

# x = [1, 2, 3, 4, 5]
# y = [6, 7, 2, 4, 5]

# p = figure(title="simple line example", x_axis_label="x", y_axis_label="y")

# p.line(x, y, legend_label="Trend", line_width=2)
# st.bokeh_chart(p, use_container_width=True)
