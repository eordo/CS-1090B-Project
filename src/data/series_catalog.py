"""
All FRED-MD and FRED-QD series used in this project.

Each dict maps the FRED series name to the tuple

    (FRED API series name, transformation code, description)

where the transformation code is for the corresponding stationarity 
transformation specified in transforms.py.
"""

# FRED-MD core series.
CORE_SERIES = {
    # GROUP 1: REAL
    # Analogous to NCDENow "Real" group.
    'INDPRO':           ('INDPRO',          5, 'Industrial Production Index'),
    'CUMFNS':           ('CUMFNS',          2, 'Capacity Utilization: Manufacturing'),
    'PAYEMS':           ('PAYEMS',          5, 'Total Nonfarm Payrolls'),
    'DPCERA3M086SBEA':  ('DPCERA3M086SBEA', 5, 'Real Personal Consumption Expenditures'),
    'CMRMTSPLx':        ('CMRMTSPL',        5, 'Real Mfg & Trade Industries Sales'),
    'AMDMNOx':          ('DGORDER',         5, 'Durable Goods Orders: Total'),
    'HOUST':            ('HOUST',           4, 'Housing Starts: Total'),
    'PERMIT':           ('PERMIT',          4, 'Building Permits: Total'),

    # GROUP 2: LABOR
    # Analogous to NCDENow "Labor" group.
    'UNRATE':           ('UNRATE',          2, 'Unemployment Rate'),
    'CLAIMSx':          ('ICSA',            5, 'Initial Jobless Claims'),
    'UEMPMEAN':         ('UEMPMEAN',        2, 'Average Duration of Unemployment (weeks)'),
    'AWHMAN':           ('AWHMAN',          1, 'Avg Weekly Hours: Manufacturing'),
    'AWOTMAN':          ('AWOTMAN',         2, 'Avg Weekly Overtime Hours: Manufacturing'),
    'CE16OV':           ('CE16OV',          5, 'Civilian Employment Level'),

    # GROUP 3: FINANCIAL
    # Analogous to NCDENow "Global" group.
    # Note: market price series (VIX, FX) are not subject to benchmark 
    # revision, so ALFRED vintage == FRED for those. They are still valid 
    # ragged-edge inputs because they have genuine publication lags relative 
    # to month-end.
    'FEDFUNDS':         ('FEDFUNDS',        2, 'Effective Federal Funds Rate'),
    'GS10':             ('GS10',            2, '10-Year Treasury Constant Maturity Rate'),
    'BAA':              ('BAA',             2, "Moody's Baa Corporate Bond Yield"),
    'T10YFFM':          ('T10YFFM',         2, '10-Year Treasury - Fed Funds Spread'),
    'BAAFFM':           ('BAAFFM',          2, 'Baa - Fed Funds Spread (credit conditions)'),
    'VXOCLSx':          ('VIXCLS',          1, 'CBOE VIX Volatility Index'),
    'EXCAUSx':          ('EXCAUS',          5, 'Canada / US Foreign Exchange Rate'),

    # GROUP 4: PRICES & SOFT
    # Analogous to NCDENow "Soft" group.
    # Known limitation vs. the NCDENow paper: FRED-MD has very limited soft 
    # data (only UMCSENTx) compared to the Korean dataset used in the paper.
    'CPIAUCSL':         ('CPIAUCSL',        6, 'CPI: All Items'),
    'CPIULFSL':         ('CPIULFSL',        6, 'CPI: All Items Less Food & Energy (Core)'),
    'PCEPI':            ('PCEPI',           6, 'PCE Chain-type Price Index'),
    'OILPRICEx':        ('MCOILWTICO',      5, 'WTI Crude Oil Price ($/barrel)'),
    'UMCSENTx':         ('UMCSENT',         2, 'U. of Michigan Consumer Sentiment'),
    'M2SL':             ('M2SL',            6, 'M2 Money Supply'),
}

# Group-series mappings.
GROUPS = {
    'REAL':        ['INDPRO', 'CUMFNS', 'PAYEMS', 'DPCERA3M086SBEA',
                    'CMRMTSPLx', 'AMDMNOx', 'HOUST', 'PERMIT'],
    'LABOR':       ['UNRATE', 'CLAIMSx', 'UEMPMEAN', 'AWHMAN', 'AWOTMAN', 'CE16OV'],
    'FINANCIAL':   ['FEDFUNDS', 'GS10', 'BAA', 'T10YFFM', 'BAAFFM', 'VXOCLSx', 'EXCAUSx'],
    'PRICES_SOFT': ['CPIAUCSL', 'CPIULFSL', 'PCEPI', 'OILPRICEx', 'UMCSENTx', 'M2SL'],
}

# FRED-QD target series.
FREDQD_SERIES = {
    # GDPC1 is the sole nowcasting target.
    # GDPDEF is retained as supplementary context but NOT used as a model input.
    'GDPC1':  ('GDPC1',  5, 'Real GDP Chained 2017$ — PRIMARY NOWCASTING TARGET'),
    'GDPDEF': ('GDPDEF', 6, 'GDP Deflator Price Index — supplementary context only'),
}
GDP_TARGET = 'GDPC1'
