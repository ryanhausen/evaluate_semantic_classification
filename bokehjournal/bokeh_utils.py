import bokeh

BOKEH_URL = 'https://cdn.pydata.org/bokeh/release/bokeh-{}.min.{}'

def get_bokeh_src():
    ver = bokeh.__version__
    link = BOKEH_URL.format(ver,'css')
    js = BOKEH_URL.format(ver, 'js')

    return link, js

def get_bokeh_widgets_src():
    ver = 'widgets-{}'.format(bokeh.__version__)
    link = LINK_STRING.format(BOKEH_URL.format(ver,'css'))
    js = JS_STRING.format(BOKEH_URL.format(ver, 'js'))

    return link, js
    
def get_bokeh_widgets_src():
    ver = 'tables-{}'.format(bokeh.__version__)
    link = LINK_STRING.format(BOKEH_URL.format(ver,'css'))
    js = JS_STRING.format(BOKEH_URL.format(ver, 'js'))

    return link, js
