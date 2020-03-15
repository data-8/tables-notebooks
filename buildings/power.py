import requests, io, datetime
import scipy
import numpy as np
import locale
from datascience import Table
from timetable import TimeTable

def high(vals):
    return np.percentile(vals, 95)
def low(vals):
    return np.percentile(vals, 5)

def loadpt(h, t1, t2, t3, t4, p_lo, p_hi):
    """ Pointwise model of trapezoidal load shape. 
    p_lo: 0-t1, Rise: t1-t2, p_hi: t2-t3, fall: t3-t4, p_lo: t4- for h in [0,24)
    """
    if h < 0 or h >= 24 :
        raise RuntimeError ('bad h to loadpt', h)
    t1 = min(max(t1, 0), 24)
    t2 = min(max(t1, t2), 24)
    t3 = min(max(t2, t3), 24)
    t4 = max(t3, t4)
    
    if t4-24 <= h and h < t1 :
        return p_lo
    elif t1 <= h and h < t2 :
        return p_lo + (p_hi - p_lo) * (h - t1)/(t2 - t1)
    elif t2 <= h and h < t3 :
        return p_hi
    elif t3 <= h and h < t4 :
        return p_hi + (p_lo - p_hi) * (h - t3)/(t4 - t3)
    elif h >= t4 :
        return p_lo
    elif t4 >= 24 and h < t4-24 :
        return p_hi + (p_lo - p_hi) * (h + 24 - t3)/(t4 - t3)
    else:
        raise RuntimeError ('bad parameters to loadpt', h)

def loadshape(hours, t1, t2, t3, t4, p_lo, p_hi):
    return [loadpt(h, t1, t2, t3, t4, p_lo, p_hi) for h in hours]

def loadmodel(t1, t2, t3, t4, p_lo, p_hi):
    return lambda h: loadpt(h, t1, t2, t3, t4, p_lo, p_hi)

def loadfit(daytable, p_lo = None, p_hi = None):
    if p_lo is None :
        p_lo = low(daytable['kW'])
    if p_hi is None :
        p_hi = high(daytable['kW'])
    params, pcov = scipy.optimize.curve_fit(loadshape, daytable['hour'], daytable['kW'], 
                                            [6,8,18,20, p_lo, p_hi], 
                                            bounds=((0,0,0,0,p_lo,p_lo), (24,24,24,24,p_hi,p_hi)))
    perr = np.sqrt(np.diag(pcov))
    return params, perr


def hour(timestamp):
    h, m = locale.atof(timestamp[0:2]), locale.atof(timestamp[3:])
    return h + m/60

class CampusPower :
    """ Interface to Brick-based energy data for a collection of buildings. """
    
    def __init__(self, host):
        self.host = host
            
    def sites(self):
        r = requests.get(self.host + '/buildings')
        if (not r.ok):
            raise Exception ('Sites status', r.status_code)
        return r.json()
    
    def classes(self):
        r = requests.get(self.host + '/classes')
        if (not r.ok):
            raise Exception ('Classes status', r.status_code)
        return r.json()
    
    def view(self, sites, classes):
        if (isinstance(sites, str)) :
            sites = [sites]
        if (isinstance(classes, str)) :
            classes = [classes]
        c = self.classes()
        s = self.sites()
        for site in sites:
            if not site in s:
                raise Exception ('Bad site: ', site)
        for cl in classes:
            if not cl in c:
                raise Exception ('Bad class: ', cl)
        return View(self.host, sites, classes)


class View :
    """ Metadata view into a power historian"""

    def __init__(self, host, sites, classes):
        self.vhost = host
        self.vsites = sites
        self.vclasses = classes
    
    def getday(self, day):
        enddate = datetime.datetime.strptime(day,"%Y-%m-%d")+datetime.timedelta(days=1)
        q = {
            "buildings": self.vsites,
            "classes": self.vclasses,
            "features": {"semester": True, "weekday": True},
            "start": day,
            "end": enddate.strftime("%Y-%m-%d")
             }
        r = requests.post(self.vhost + '/data', json = q)
        if (not r.ok):
            raise Exception ('getday request', r.status_code)
        df = Table.read_table(io.StringIO(r.text))
        units = df['units'][0]
        metadata = {'units'    : units,
                    'semester' : df['semester'][0],
                    'daytype'  : df['weekday'][0],
                    'day'      : day,
                    'sites'    : self.vsites,
                    'classes'  : self.vclasses}
        timeseries = df.select(['time', 'value']).group('time', sum).relabel('value sum', units)
        timeseries['hour'] = timeseries.apply(hour, 'time')
        timeseries.move_to_start('hour')
        return timeseries.drop('time'), metadata
    
    def getdays(self, start_day, end_day):
        q = {
            "buildings": self.vsites,
            "classes": self.vclasses,
            "features": {},
            "start": start_day,
            "end": end_day
             }
        r = requests.post(self.vhost + '/data', json = q)
        if (not r.ok):
            raise Exception ('getday request', r.status_code)
        df = Table.read_table(io.StringIO(r.text))
        units = df['units'][0]
        metadata = {'units'    : units,
                    'start'    : start_day,
                    'end'      : end_day,
                    'sites'    : self.vsites,
                    'classes'  : self.vclasses}
        timeseries = df.select(['date', 'time', 'value', 'id']).pivot('date', 'time', 'value', sum)
        timeseries['hour'] = timeseries.apply(hour, 'time')
        timeseries.move_to_start('hour')
        return timeseries.drop('time'), metadata

    def model_days(self, start, end):
        date = datetime.datetime.strptime(start,"%Y-%m-%d")
        enddate = datetime.datetime.strptime(end,"%Y-%m-%d")
        site_history = Table(['day', 'daytype', 'term', 'p_5', 'p_ave', 'p_95', 't1', 't2', 't3', 't4', 'p_lo', 'p_hi'])
        while date != enddate :
            day = datetime.datetime.strftime(date, "%Y-%m-%d")
            try :
                ts, md = self.getday(day)
                p_5 = np.percentile(ts['kW'], 5)
                p_ave = np.mean(ts['kW'])
                p_95 = np.percentile(ts['kW'], 95)
                print(day, md['daytype'], p_5, p_ave, p_95)
                params, perr = loadfit(ts, p_lo = p_5, p_hi = p_95)
                row = (day, md['daytype'], md['semester'], p_5, p_ave, p_95, *params)
                site_history.append(row)
            except :
                print("failed to get", day)
            date =  date + datetime.timedelta(days=1)
        return site_history
