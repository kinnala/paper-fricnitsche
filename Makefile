default:
	ipython test3_fixed_terms.py > uniform.csv
	ipython test3_adaptive_fixed_terms.py > adaptive.csv
	ipython test3_plot.py

export:
	mkdir -p export
	pdfcrop {,export/}test3_adaptive_defo_21.pdf
	pdfcrop {,export/}test3_adaptive_lambdan_21.pdf
	pdfcrop {,export/}test3_adaptive_lambdat_21.pdf
	pdfcrop {,export/}test3_adaptive_un_21.pdf
	pdfcrop {,export/}test3_adaptive_ut_21.pdf
	pdfcrop {,export/}test3_uniform_defo_4.pdf
	pdfcrop {,export/}test3_uniform_lambdan_4.pdf
	pdfcrop {,export/}test3_uniform_lambdat_4.pdf
	pdfcrop {,export/}test3_uniform_un_4.pdf
	pdfcrop {,export/}test3_uniform_ut_4.pdf
	pdfcrop {,export/}test3_plot_convergence.pdf
	pdfcrop {,export/}test3_uniform_contact_convergence.pdf
	pdfcrop {,export/}test3_adaptive_contact_convergence.pdf
	pdfcrop {,export/}test3_adaptive_estimator_split.pdf

clean:
	rm -r export/
