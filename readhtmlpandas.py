# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 19:21:15 2021

@author: IKM1YH
"""

import pandas as pd
url_str = 'Phase 3 reporting - PS-EC Software PMT (Process, Method & Tool Development) Wiki.html'


df_lst = pd.read_html(url_str, header=0)