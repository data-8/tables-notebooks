from datascience import Table
import numpy as np
import matplotlib.pyplot as plots
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import linear_model

class ML_Table(Table):
    """Table with ML operators defined"""

    def __init__(self, *args, **kwargs):
        Table.__init__(self, *args, **kwargs)
            
    @classmethod
    def from_table(cls, tbl):
        ml_tbl = ML_Table()
        for label in tbl.labels:
            ml_tbl[label] = tbl[label]
        return ml_tbl
    
    # Column generators
    @classmethod
    def rnorm(cls, label, n, seed=None, mean=0, sd=1):
        if seed is not None:
            np.random.seed(seed)
        return ML_Table().with_column(label, np.random.normal(loc=mean, scale=sd, size=n))

    @classmethod
    def sequence(cls, label, n, low=0, high=1):
        return ML_Table().with_column(label, np.arange(low, high, (high-low)/n))
          
    def summary(self, ops=None):
        def FirstQu(x):
            return np.percentile(x, 25)
        def ThirdQu(x):
            return np.percentile(x, 5)
        if ops is None:
            ops=[min, FirstQu, np.median, np.mean, ThirdQu, max]
        return self.stats(ops=ops)

    # Common statistical machine learning operators
    def regression_1d(self, x_label, Y_label):
        """Return a function that is a linear model of f(x) = Y."""
        m, b = np.polyfit(self[x_label], self[Y_label], 1)
        return lambda x: m*x + b

    def regression_params(self, output_label):
        """Form a model of a table using linear regression and return as a function."""
        input_labels = [lbl for lbl in self.labels if not lbl == output_label]
        reg = linear_model.LinearRegression()
        input_vectors = [self[lbl] for lbl in input_labels]
        reg.fit(np.transpose(input_vectors), self[output_label])
        return reg.coef_, reg.intercept_

    def ridge_params(self, output_label, **kwargs):
        """Form a model of a table using linear regression and return as a function."""
        input_labels = [lbl for lbl in self.labels if not lbl == output_label]
        reg = linear_model.Ridge(**kwargs)
        input_vectors = [self[lbl] for lbl in input_labels]
        reg.fit(np.transpose(input_vectors), self[output_label])
        return reg.coef_, reg.intercept_

    def regression(self, output_label, method=None, **kwargs):
        """Make a function that is the model over input values using ordinary lienar regression."""
        if method is None:
            method = self.regression_params
        m, b = method(output_label, **kwargs)
        def _reg_fun(*args):
            psum = b
            for p,v in zip(m, args):
                psum += p*v
            return psum
        return _reg_fun

    def ridge(self, output_label,  **kwargs):
        return self.regression(output_label, method=self.ridge_params, **kwargs)

    def mse(self, y_label, f_label):
        """Calulate the mean square error of a observations y from estimate f."""
        return sum((self[y_label]-self[f_label])**2)/self.num_rows
    
    # Visualization

    def boxplot(self, column_for_xcats=None, select=None, height=4, width=6,  **vargs):
        """Plot box-plots for the columns in a table.

        If no column for categories is specified, 
        a boxplot is produced for each column (or for the columns designated
        by `select`) labeled by the column name.
        If one is satisfied, a box plot is produced for each other column
        using a pivot on the categories.

        Every selected column must be numerical, other than the category column

        Args:

        Kwargs:
            select (column name or list): columns to include

            vargs: Additional arguments that get passed into `plt.plot`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
                for additional arguments that can be passed into vargs.
        """
        options = self.default_options.copy()
        options.update(vargs)

        x_labels = self.labels
        if select is not None:
            x_labels = self._as_labels(select)

        if column_for_xcats is None:
            fig, ax = plots.subplots(figsize=(width,height))
            data = [self[lbl] for lbl in x_labels]
            ax.boxplot(data, labels=x_labels, **vargs)
            return ax
        else:
            grouped = self.select(x_labels).group(column_for_xcats, collect=lambda x:x)
            x_labels = [lbl for lbl in x_labels if lbl != column_for_xcats]
            fig, axes = plots.subplots(len(x_labels), 1, figsize=(width, height))
            if len(x_labels) == 1:
                axes = [axes]
            for (lbl,axis) in zip(x_labels, axes):
                axis.boxplot(grouped[lbl], labels=grouped[column_for_xcats])
                axis.set_ylabel(lbl)
            if len(x_labels) == 1:
                return axes[0]
            else:
                return axes

    def plot_fit_1d(self, x_label, y_label, model_fun):
        """Visualize the error in f(x) = y + error."""
        fig, ax = plots.subplots()
        ax.scatter(self[x_label], self[y_label])
        f_tbl = self.select([x_label, y_label]).sort(x_label, descending=False)
        fun_x = f_tbl.apply(model_fun, x_label)
        ax.plot(f_tbl[x_label], fun_x)
        for i in range(f_tbl.num_rows):
            ax.plot([f_tbl[x_label][i], f_tbl[x_label][i]], 
                    [fun_x[i], f_tbl[y_label][i] ], 'r-')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax

    def plot_fit_2d(self, x_label, y_label, z_label, model_fun=None, n_mesh=50, 
                    xmin=None, xmax=None, ymin=None, ymax=None,
                    rstride=5, cstride=5, width=6, height=4):
        fig = plots.figure(figsize=(width, height))
        ax = fig.add_subplot(111, projection='3d')
        if model_fun is not None:
            if xmin is None:
                xmin = min(self[x_label])
            if xmax is None:
                xmax = max(self[x_label])
            if ymin is None:
                ymin = min(self[y_label])
            if ymax is None:
                ymax = max(self[y_label])
            xstep = (xmax-xmin)/n_mesh
            ystep = (ymax-ymin)/n_mesh
            xv = np.arange(xmin, xmax + xstep, xstep)
            yv = np.arange(ymin, ymax + ystep, ystep)
            X, Y = np.meshgrid(xv, yv)
            Z = model_fun(X, Y)
            ax.plot_surface(X, Y, Z, rstride=5, cstride=5, linewidth=1, cmap=cm.coolwarm)
            ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=1, color='b')
            for (x, y, z) in zip(self[x_label], self[y_label], self[z_label]):
                mz = model_fun(x,y)
                ax.plot([x,x], [y,y], [z,mz], color='black')

        ax.scatter(self[x_label], self[y_label], self[z_label], c='r', marker='o')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        return ax

    def plot_fit(self, f_label, model_fun, width=6, height=4):
        """Visualize the goodness of fit of a model."""
        labels = [lbl for lbl in self.labels if not lbl == f_label]
        assert len(labels) <= 2, "Too many dimensions to plot"
        if len(labels) == 1:
            return self.plot_fit_1d(labels[0], f_label, model_fun)
        else:
            return self.plot_fit_2d(labels[0], labels[1], f_label, model_fun, width=width, height=height)
                    
    # Superclass methods that return a new object that must be coerced
    def table_apply(self, method, *args, **kwargs):
        tbl = method(self, *args, **kwargs)
        if isinstance(tbl, Table):
            return ML_Table.from_table(tbl)
        else:
            return tbl
    
    def select(self, *args, **kwargs):
        return self.table_apply(Table.select, *args, **kwargs)
    
    def group(self, *args, **kwargs):
        return self.table_apply(Table.group, *args, **kwargs)
    
    def groups(self, *args, **kwargs):
        return self.table_apply(Table.groups, *args, **kwargs)
    
    def where(self, *args, **kwargs):
        return self.table_apply(Table.where, *args, **kwargs)
    
    def copy(self, *args, **kwargs):
        return self.table_apply(Table.copy, *args, **kwargs)
    
    def sort(self, *args, **kwargs):
        return self.table_apply(Table.sort, *args, **kwargs)
    
    def take(self, *args, **kwargs):
        return self.table_apply(Table.take, *args, **kwargs)
    
    def exclude(self, *args, **kwargs):
        return self.table_apply(Table.exclude, *args, **kwargs)
    
