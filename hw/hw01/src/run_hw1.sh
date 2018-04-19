#!/usr/env bash

x0=10
x1=$x0

c=1
python3 line_search_zoom.py $c $x0 $x1

c=10
python3 line_search_zoom.py $c $x0 $x1
