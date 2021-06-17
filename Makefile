default:
	ipython test3_fixed_terms.py > uniform.csv
	ipython test3_adaptive_fixed_terms.py > adaptive.csv
	ipython test3_plot.py

export:
	mkdir -p export
	cp test3_adaptive_defo_20.pdf export
	cp test3_adaptive_lambdan_20.pdf export
	cp test3_adaptive_lambdat_20.pdf export
	cp test3_uniform_defo_4.pdf export
	cp test3_uniform_lambdan_4.pdf export
	cp test3_uniform_lambdat_4.pdf export
	cp test3_plot_convergence.pdf export

clean:
	rm -r export/
