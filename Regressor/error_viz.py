import pandas as pd
import requests
import io
import numpy as np

import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings("ignore")

# Error Viz Class for Regression


class ErrorVizRegression:
    def __init__(self, df, modelobj, target, model_df=None, groupbyvars=None):
        self.df = df
        self.modelobj = modelobj
        if model_df is None:
            self.model_df = pd.get_dummies(df)
        else:
            self.model_df = model_df
        self.target = target
        self.groupbyvars = groupbyvars

    def _transform(self, df, col, pred):

        """
        Function to transform error values according to percentile grouping.

        """

        errs = df["err"].values
        agg_err = pd.DataFrame(
            {
                col: df[col].mode(),
                pred: np.nanmedian(df[pred]),
                "errPos": np.nanmedian(errs[errs >= 0]),
                "errNeg": np.nanmedian(errs[errs <= 0]),
            },
            index=[0],
        )

        return agg_err

    def _convert_categorical_independent(self, df):

        """
        utility function to convert pandas dtypes 'categorical'
        into numerical columns
        :param dataframe: dataframe to perform adjustment on
        :return: dataframe that has converted strings to numbers
        :rtype: pd.DataFrame

        """
        # we want to change the data, not copy and change
        dataframe = df.copy(deep=True)
        # convert all strings to categories and format codes
        for str_col in dataframe.select_dtypes(include=["O", "category"]):
            dataframe.loc[:, str_col] = pd.Categorical(dataframe.loc[:, str_col])
        # convert all category datatypes into numericf
        cats = dataframe.select_dtypes(include=["category"])
        # warn user if no categorical variables detected
        if cats.shape[1] == 0:
            warnings.warn("Pandas categorical variable types not detected", UserWarning)
        # iterate over these columns
        for category in cats.columns:
            dataframe.loc[:, category] = dataframe.loc[:, category].cat.codes

        return dataframe

    def create_plot_continuous(
        self, feature, groupbyvar=None, groupbyvalue=None, fill_nulls=True
    ):

        """
        Creates error plot for continuous feature.

        Params:-

        feature: Feature for which plot is to be made
        groupbyvar: Categorical feature in dataset for grouping
        groupbyvalue: Value of given categorical feature to be grouped by
        fill_nulls: Bool value to enable or disable filling null error values in plot dataframe with 0

        Returns: Plotly graph object instance

        """

        # For groupbyvars make plot dataframe accordingly
        if groupbyvar is not None:
            plot_title = groupbyvar + ": " + groupbyvalue
            gb_title = groupbyvar + "_" + groupbyvalue

            # Make predictions for groupby value constraint
            pred = self.modelobj.predict(self.model_df[self.model_df[gb_title] == 1])

            plot_df = pd.DataFrame(
                {
                    "xaxis": self.model_df[self.model_df[gb_title] == 1][
                        feature
                    ].sort_values(),
                    "pred": pred,
                    "true": self.df[self.df[groupbyvar] == groupbyvalue][self.target],
                }
            )

        elif groupbyvar is None:
            plot_title = feature
            pred = self.modelobj.predict(self.model_df)

            plot_df = pd.DataFrame(
                {
                    "xaxis": self.model_df[feature].sort_values(),
                    "pred": pred,
                    "true": self.df[self.target],
                }
            )

        # Calculate Errors
        plot_df["err"] = plot_df["pred"] - plot_df["true"]

        # Order by Bins
        plot_df["bins"] = pd.qcut(
            plot_df["xaxis"].sort_values(), q=100, labels=False, duplicates="drop"
        )

        # Replace values by max value in every bin
        maxvals = plot_df.groupby("bins")["xaxis"].max().reset_index(name="maxcol")
        plot_df = plot_df.join(
            maxvals, on="bins", how="inner", lsuffix="_left", rsuffix="_right"
        ).rename(columns={"bins_left": "bins"})
        # drop and rename columns
        plot_df = plot_df.drop(["bins", "xaxis"], axis=1).rename(
            columns={"maxcol": "xaxis"}
        )

        # Apply transform function to calculate medians of prediction,positive and negative errors
        plot_df = plot_df.groupby(["xaxis"], as_index=False).apply(
            self._transform, col="xaxis", pred="pred"
        )

        # For null values fill with 0
        if fill_nulls is True:
            fillvalues = {"errPos": 0, "errNeg": 0}
            plot_df.fillna(value=fillvalues, inplace=True)

        smooth = "spline"

        ePlot = make_subplots(rows=2, cols=1, vertical_spacing=0)

        # Add traces for medians of prediction, positive and negative error
        ePlot.add_trace(
            go.Scatter(
                x=plot_df["xaxis"],
                y=plot_df["errPos"] + plot_df["pred"],
                # name = 'Median Positive Error',
                showlegend=False,
                mode="lines",
                # connectgaps = True,
                line_shape=smooth,
                line=dict(shape="linear", width=0.9, color="lightgray"),
            ),
            row=1,
            col=1,
        )
        ePlot.add_trace(
            go.Scatter(
                x=plot_df["xaxis"],
                y=plot_df["errNeg"] + plot_df["pred"],
                # name = 'Median Negative Error',
                showlegend=False,
                mode="lines",
                line_shape=smooth,
                line=dict(shape="linear", width=0.9, color="lightgray"),
                # connectgaps = True,
                fill="tonexty",
                fillcolor="rgba(68, 68, 68, 0.3)",
            ),
            row=1,
            col=1,
        )

        ePlot.add_trace(
            go.Scatter(
                x=plot_df["xaxis"],
                y=plot_df["pred"],
                line_shape=smooth,
                mode="lines",
                # line = dict(color = '#099A67'),
                name=plot_title,
            ),
            row=1,
            col=1,
        )

        # Add rug for feature
        ePlot.add_trace(
            go.Box(
                x=plot_df["xaxis"],
                marker_symbol="line-ns-open",
                marker_color="blue",
                boxpoints="all",
                jitter=0,
                fillcolor="rgba(255,255,255,0)",
                line_color="rgba(255,255,255,0)",
                hoveron="points",
                name="Rug for " + plot_title,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        ePlot.update_layout(  # title = 'grouped by medium alcohol',
            xaxis_title=feature,
            xaxis={"side": "top"},  # xaxis2 = dict(tickmode = 'linear', tick0 = 0),
            yaxis_title=self.target,
            yaxis=dict(domain=[0.15, 1]),
            yaxis2=dict(domain=[0, 0.15], showticklabels=False),
            margin=dict(l=30, t=30, b=30),
        )

        return ePlot

    def create_plot_categorical(self, feature, groupbyvar=None, groupbyvalue=None):
        """
        Creates error plot for categorical feature.

        Params:-

        feature: Feature for which plot is to be made
        groupbyvar: Categorical feature in dataset for grouping
        groupbyvalue: Value of given categorical feature to be grouped by

        Returns: Plotly graph object instance

        """

        if groupbyvar is not None:

            plot_name = feature + " for " + groupbyvar + ": " + groupbyvalue
            gb_title = groupbyvar + "_" + groupbyvalue
            # Make predictions for groupby value constraint
            pred = self.modelobj.predict(self.model_df[self.model_df[gb_title] == 1])

            plot_df = pd.DataFrame(
                {
                    "xaxis": self.model_df[self.model_df[gb_title] == 1][feature],
                    "pred": pred,
                    "true": self.df[self.df[groupbyvar] == groupbyvalue][self.target],
                }
            )

        elif groupbyvar is None:
            pred = self.modelobj.predict(self.model_df)

            plot_df = pd.DataFrame(
                {"xaxis": self.df[feature], "pred": pred, "true": self.df[self.target]}
            )

            plot_name = feature

        # Calculate Errors
        plot_df["err"] = plot_df["true"] - plot_df["pred"]

        # Apply transform function to calculate median prediction, positive and negative errors
        plot_df = plot_df.groupby(["xaxis"], as_index=False).apply(
            self._transform, col="xaxis", pred="pred"
        )

        catplot = go.Figure()
        catplot.add_trace(
            go.Scatter(
                x=plot_df["xaxis"],
                y=plot_df["pred"],
                mode="markers",
                marker=dict(size=10),
                error_y=dict(
                    # type = 'data',
                    arrayminus=plot_df["errPos"],
                    array=-1 * plot_df["errNeg"],
                    width=25,
                ),
                name=feature,
            )
        )
        catplot.update_layout(
            yaxis=dict(tickmode="linear", dtick=0.2),
            xaxis_title=feature,
            yaxis_title=self.target,
            margin=dict(t=30, l=30, b=30),
        )
        catplot.update_xaxes(type="category")

        return catplot

    def _cont_groupby_dropdown(self, groupbyvar):

        """
        Function to create updatemenus attribute for making visibility dropdown based on groupby values.

        Returns: updatemenus list for passing through update_layout()

        """
        # Create dictionary for visibility vectors for each dropdown option
        vis_dict = dict()
        uq = self.df[groupbyvar].unique()
        for i in range(len(uq)):
            fVec = [False] * (len(uq) * 4)
            vec = [True] * 4
            fVec[i * 4 : i * 4 + len(vec)] = vec
            vis_dict[uq[i]] = fVec

        # Create first button for all groupby values
        buttons = [
            dict(
                label="All",
                method="update",
                args=[
                    {"visible": [True for i in range(len(uq) * 4)]},
                    {"title": "All"},
                ],
            )
        ]

        # Create buttons for each groupby value
        for i in range(len(uq)):
            buttons.append(
                dict(
                    label=uq[i],
                    method="update",
                    args=[{"visible": vis_dict[uq[i]]}, {"title": uq[i]}],
                )
            )

        updatemenus = list(
            [
                dict(
                    active=0,
                    buttons=buttons,
                )
            ]
        )

        return updatemenus

    def _cat_groupby_dropdown(
        self,
        figdata,
        groupbyvar,
        feature=None,
        tickvals_all=None,
        tickvals_individual=None,
    ):

        """
        Function to create updatemenus attribute for making visibility dropdown based on groupby values.

        Returns: updatemenus list for passing through update_layout()

        """
        # Create dictionary for visibility vectors for each dropdown option
        vis_dict = dict()
        uq = self.df[groupbyvar].unique()
        for i in range(len(uq)):
            fVec = [False] * (len(uq))
            fVec[i] = True
            vis_dict[uq[i]] = fVec

        x_dict = dict(zip(tickvals_all, self.df[feature].unique()))

        ticktext = np.sort(self.df[feature].unique())

        buttons = [
            dict(
                label="All",
                method="update",
                args=[
                    {"visible": [True for i in range(len(uq))]},
                    {
                        "title": "All",
                        "xaxis": dict(
                            title=feature,
                            tickmode="array",
                            tickvals=tickvals_all,
                            ticktext=ticktext,
                        ),
                    },
                ],
            )
        ]

        for i in range(len(uq)):
            x_dict2 = dict(
                zip(figdata[i]["x"], np.sort(self.df[feature].unique()).tolist())
            )
            buttons.append(
                dict(
                    label=uq[i],
                    method="update",
                    args=[
                        {"visible": vis_dict[uq[i]]},
                        {
                            "title": uq[i],
                            "xaxis": dict(
                                title=feature,
                                tickmode="array",
                                tickvals=figdata[i]["x"],
                                ticktext=[x_dict2[k] for k in figdata[i]["x"]],
                            ),
                        },
                    ],
                )
            )
        updatemenus = list(
            [
                dict(
                    active=0,
                    showactive=True,
                    buttons=buttons,
                )
            ]
        )

        return updatemenus

    def create_combined_cont_plot(self, feature, groupbyvar=None, fill_nulls=True):

        """
        Function to create combined error plot of a continous variable with grouping.

        Params:-
        feature: Feature for which plot is to be made
        groupbyvar: Variable for grouping if any
        fill_nulls: Bool value to enable or disable filling null error values in plot dataframe with 0

        Returns: plotly graph object instance

        """

        figures = []

        if groupbyvar is not None:
            groupbyvalues = self.df[groupbyvar].unique()

            # Create plots for every combination and store in a list
            for gbvalue in groupbyvalues:
                newfig = self.create_plot_continuous(
                    feature=feature,
                    groupbyvar=groupbyvar,
                    groupbyvalue=gbvalue,
                    fill_nulls=fill_nulls,
                )
                figures.append(newfig)

        else:
            # For no grouping, return simple continuous error plot
            newfig = self.create_plot_continuous(feature=feature)
            # figures.append(newfig)
            return newfig

        fig_data = [fig.data for fig in figures]

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0)
        for i in range(len(fig_data)):
            for j in range(len(fig_data[i])):
                # Pass row = 1 for traces
                row = 1
                # Pass specific row (2) for Raster Plot
                if j == len(fig_data[i]) - 1:
                    row = 2
                fig.add_trace(fig_data[i][j], row=row, col=1)

        fig.update_layout(  # title = 'grouped by medium alcohol',
            xaxis_title=feature,
            xaxis={"side": "top"},
            yaxis_title=self.target,
            margin=dict(l=30, t=50, b=30),
        )

        if groupbyvar is not None:
            # Add dropdown if grouping is enabled
            fig.update_layout(
                updatemenus=self._cont_groupby_dropdown(groupbyvar),
                # xaxis2 = dict(tickmode = 'auto', tick0 = 0, dtick = 10),
                yaxis=dict(domain=[0.15, 1]),
                yaxis2=dict(domain=[0, 0.15], showticklabels=False),
            )

        return fig

    def create_combined_cat_plot(self, feature, groupbyvar=None):

        """
        Function to create combined error plot of a categorical variable with or without grouping.

        Params:-
        feature: Feature for which plot is to be made
        groupbyvar: Variable for grouping if any

        Returns: plotly graph object instance

        """
        if groupbyvar is None:

            # Return simple categorical plot if no grouping
            fig = self.create_plot_categorical(feature)

        elif groupbyvar is not None:

            groupbyvalues = self.df[groupbyvar].unique()
            categories = self.df[feature].unique()

            # Convert categorical columns to numeric
            newdf = self._convert_categorical_independent(self.df)
            newmodel_df = pd.get_dummies(newdf.loc[:, newdf.columns != self.target])

            new_categories = newdf[feature].unique()
            newgbvalues = newdf[groupbyvar].unique()

            fig = go.Figure()

            for i in range(len(groupbyvalues)):
                gb_title = groupbyvar + "_" + groupbyvalues[i]
                pred = self.modelobj.predict(
                    self.model_df[self.model_df[gb_title] == 1]
                )
                plot_df = pd.DataFrame(
                    {
                        "xaxis": newdf[newdf[groupbyvar] == newgbvalues[i]][feature]
                        * 1.5,  # Multiplied by 1.5 to facilitate spacing between categories
                        "pred": pred,
                        "true": newdf[newdf[groupbyvar] == newgbvalues[i]][self.target],
                    }
                )

                # Calculate error
                plot_df["err"] = plot_df["true"] - plot_df["pred"]

                # Apply transform function to calculate median prediction, positive and negative errors
                plot_df = plot_df.groupby(["xaxis"], as_index=False).apply(
                    self._transform, col="xaxis", pred="pred"
                )

                fig.add_trace(
                    go.Scatter(
                        x=plot_df["xaxis"] + (0.25 * i),
                        y=plot_df[
                            "pred"
                        ],  # Multiply x values by 0.25*i to facilitate shifting of plots for different groups
                        name=groupbyvalues[i],
                        mode="markers",
                        marker=dict(size=10),
                        error_y=dict(
                            type="data",
                            arrayminus=plot_df["errPos"],
                            array=-1 * plot_df["errNeg"],
                            width=25,
                        ),
                    )
                )

        if groupbyvar is not None:

            # Adjust X tick values and labels

            tickvals_all = (new_categories + 0.25 * (len(newgbvalues))) / 2
            # tickvals.sort()

            figdata = fig.data
            fig.update_layout(
                updatemenus=self._cat_groupby_dropdown(
                    groupbyvar=groupbyvar,
                    figdata=figdata,
                    feature=feature,
                    tickvals_all=tickvals_all,
                    tickvals_individual=new_categories,
                ),
                xaxis_title=feature,
                yaxis_title=self.target,
                margin=dict(l=30, t=30, b=30),
            )

        return fig


#    def create_err_plot(self):
