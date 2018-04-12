from journal import Journal
from bokeh.plotting import figure

journal = Journal('Example Output')

journal.append_h1('This Is An Example!')

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
p.line(x, y, legend="Temp.", line_width=2)
journal.append_bokeh(p)

p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
p.line(x, y, legend="Temp.", line_width=2)
journal.append_bokeh(p)

journal.show()

