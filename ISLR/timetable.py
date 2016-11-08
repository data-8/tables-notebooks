from datascience import Table
import numpy as np
from scipy.interpolate import UnivariateSpline

class TimeTable(Table):
    """Table with a designated column as a sequence of times in the first column."""

    def __init__(self, *args, time_column = 'Year'):
        Table.__init__(self, *args)
        self.time_column = time_column
    
    def clone_bare(self):
        return TimeTable(time_column = self.time_column)
    
    def clone_time(self):
        return self.clone_bare().with_column(self.time_column, self[self.time_column])
            
    @classmethod
    def from_table(cls, tbl, time_col):
        ttbl = cls(time_column = time_col)
        for label in tbl.labels:
            ttbl[label] = tbl[label]
        return ttbl
    
    def __get_attr__(self, name):
        def wrapper(*args, **kwargs):
            # Wrap superclass method to coerce result back to TimeTable
            tbl = self.name(*args, **kwargs)
            if isinstance(tbl, Table) and self.time_column in tbl.labels:
                return TimeTable.from_table(tbl, self.time_column)
            else:
                return tbl
        print("Get TimeTable Attr", name)
        if hasattr(Table, name):
            return wrapper
        else:
            raise AttributeError
            
    @classmethod
    def by_time(cls, tbl, time_col, category_col, collect_col):
        """Construct a time table by aggregating rows of each category by year."""
        tbl_by_year = tbl.select([category_col, time_col, collect_col]).pivot(category_col, time_col, collect_col, collect=sum)
        return cls(tbl_by_year.labels, time_column=time_col).append(tbl_by_year)
    
    @property
    def categories(self):
        return [label for label in self.labels if label != self.time_column]
    
    # Table transformation methods that preserve TimeTable behavior
    
    def table_apply(self, method, *args, **kwargs):
        tbl = method(self, *args, **kwargs)
        if self.time_column in tbl.labels:
            return TimeTable.from_table(tbl, self.time_column)
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
    
    # TimeTable methods utilizing time_column

    def order_cols(self):
        """Create a TimeTable with categories ordered by the values in last row."""
        def col_key(label):
            return self.row(self.num_rows-1)[self.labels.index(label)]
        order = sorted(self.categories, key=col_key, reverse=True)
        tbl = self.copy()
        for label in order:
            tbl.move_to_end(label)
        return tbl
    
    def oplot(self):
        return self.order_cols().plot(self.time_column)
    
    def top(self, n):
        """Create a new TimeTable containing the n largest columns."""
        ttbl = self.order_cols()
        return ttbl.select(ttbl.labels[0:n+1])
    
    def sum_rows(self):
        """Sum the rows in a TimeTable besides the time column."""
        tbl = self.drop(self.time_column)
        return [sum(row) for row in tbl.rows]
    
    def ratio(self, tbl_denom):
        """Create ratio of a TimeTable to a matching one."""
        rtbl = TimeTable(time_column = self.time_column).with_column(self.time_column, self[self.time_column])
        for label in self.categories:
            rtbl[label] = self[label] / tbl_denom[label]
        return rtbl
    
    def delta(self):
        """Construct a TimeTableable of successive differences down each non-time column."""
        delta_tbl = self.clone_bare()
        delta_tbl[self.time_column] = self[self.time_column][1:]
        for col in self.categories:
            delta_tbl[col] = self[col][1:] - self[col][:-1]
        return delta_tbl
    
    def fill(self, interval=1):
        times = [t for t in np.arange(self[self.time_column][0], self[self.time_column][-1] + interval, interval)]
        ftbl = TimeTable(time_column = self.time_column).with_column(self.time_column, times)
        for col in self.categories:
            spl = UnivariateSpline(self[self.time_column], self[col])
            ftbl[col] = spl(times)
        return ftbl
    
    def interp(self, interval=1):
        times = [t for t in np.arange(self[self.time_column][0], self[self.time_column][-1] + interval, interval)]
        ftbl = TimeTable(time_column = self.time_column).with_column(self.time_column, times)
        for col in self.categories:
            ftbl[col] = np.interp(times, self[self.time_column], self[col])
        return ftbl

    def rel_delta(self):
        """Construct a TimeTableable of successive differences down each non-time column."""
        delta_tbl = self.clone_bare()
        delta_tbl[self.time_column] = self[self.time_column][1:]
        time_delta = self[self.time_column][1:] - self[self.time_column][:-1]
        for col in self.categories:
            delta_tbl[col] = (1+(self[col][1:] - self[col][:-1])/self[col][:-1])/time_delta
        return delta_tbl
    
    def norm_by_row(self, base_row=0):
        """Normalize columns of a TimeTable by a row"""
        normed_tbl = self.clone_time()
        for label in self.categories:
            normed_tbl[label] = self[label]/self[label][base_row]
        return normed_tbl
    
    def norm_by_time(self, time):
        return self.norm_by_row(np.where(self[self.time_column] == time)[0][0])
    
    def sum_cols(self):
        """Sum the columns of TimeTable."""
        csum = 0
        for c in self.categories:
            csum += self[c]
        return csum
    
    def fraction_cols(self):
        """Convert each column to a fraction by row."""
        total = self.sum_cols()
        ftbl = self.clone_time()
        for lbl in self.categories:
            ftbl[lbl] = self[lbl]/total
        return ftbl
    
    def forecast_table(self, past, ahead, inc=1):
        """Project a TimeTable forward.  inc must match the interval"""
        last_time = self[self.time_column][-1]
        past_times = self[self.time_column][-past-1:-1]
        fore_time = np.arange(last_time + inc, last_time + inc + ahead, inc)
        def project(lbl):
            m, b = np.polyfit(past_times, self[lbl][-past-1:-1], 1)
            return [m*time + b for time in fore_time]
        xtbl = Table().with_columns([(self.time_column, fore_time)] + [(label, project(label)) for label in self.categories])
        return self.copy().append(xtbl)
    
    def extend_table(self, ahead, inc=1):
        """Project a TimeTable forward from last interval.  inc must match the interval"""
        last_time = self[self.time_column][-1]
        fore_time = np.arange(last_time + inc, last_time + inc + ahead, inc)
        def project(lbl):
            b = self[lbl][-1]
            m = self[lbl][-1] - self[lbl][-2]
            return [m*(time+1)*inc + b for time in range(ahead)]
                                             
        xtbl = Table().with_columns([(self.time_column, fore_time)] + [(label, project(label)) for label in self.categories])
        return self.copy().append(xtbl)