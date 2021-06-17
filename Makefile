default:
	ipython test3_fixed_terms.py > uniform.csv
	ipython test3_adaptive_fixed_terms.py > adaptive.csv
	ipython test3_plot.py
