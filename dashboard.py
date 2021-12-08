from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.datastructures import ImmutableMultiDict
from flask_login import login_required, current_user
from datetime import datetime
from PIL import Image
from bokeh.models import ColumnDataSource, Select, Slider, Range1d
from bokeh.models.callbacks import CustomJS
from bokeh.resources import INLINE, CDN
from bokeh.embed import components, file_html, json_item
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row

import mysql.connector
import pandas as pd
import secrets
import store
import json
import os


dashboard = Blueprint('dashboard', __name__)

from app import db, app
from models import UpdateUserInfo


def connect():
    _db = mysql.connector.connect(user=store.user, password=store.password,
                                  host=store.hostname,
                                  database='dashboard')
    return _db


def query_db(query):
    _db = connect()
    data = pd.read_sql(query, _db)
    _db.close()
    return data


def get_filter_data():
    query = """SELECT DISTINCT p.category, p.brand, p.occasion, p.collection, p.prod_material
               FROM  product p"""
    return query_db(query)


@dashboard.route('/dashboard/summary')
@login_required
def summary():
    plot = figure()
    plot.circle([1,2], [3,4])

    data = {'filter': get_filter_data()}
    return render_template('dashboard/dashboard.html', data=data)


@dashboard.route('/dashboard/sales')
@login_required
def sales():
    query = """SELECT p.category, p.brand, p.occasion, p.collection, s.sales, d.month, d.quarter 
                FROM sales s
                LEFT JOIN product p ON s.product_id = p.id
                LEFT JOIN date d on s.date_id = d.id"""
    sales_data = query_db(query)
    data = {'filter': get_filter_data()}

    title = "Sales by Month"
    monthly_sales = sales_data.groupby('month').sales.sum().reset_index()
    monthly_sales.month = monthly_sales.apply(lambda x: datetime.strptime(x.month, "%b").month, axis=1)
    monthly_sales = monthly_sales.sort_values(by='month')

    d = monthly_sales.values.tolist()
    c = monthly_sales.columns.tolist()
    d.insert(0, c)
    temp_data = json.dumps({'title': title, 'data': d})

    return render_template('dashboard/sales.html', data=data, tempdata=temp_data)


@dashboard.route('/dashboard/availability')
@login_required
def availability():
    data = {'filter': get_filter_data()}
    return render_template('dashboard/availability.html', data=data)


@dashboard.route('/dashboard/promotions')
@login_required
def promotions():
    data = {'filter': get_filter_data()}
    return render_template('dashboard/promotions.html', data=data)


@dashboard.route('/dashboard/reviews')
@login_required
def reviews():
    data = {'filter': get_filter_data()}
    return render_template('dashboard/reviews.html', data=data)

@dashboard.route('/dashboard/products', methods=['GET', 'POST'])
@login_required
def products():
    if request.method == 'POST':
        id = request.form['prod_id']
        return redirect(url_for('dashboard.product_details', id=id))

    query = """
    SELECT p.id, p.title, p.image_url, p.details_url, p.category, p.prod_material, s.sales
    FROM sales s
    LEFT  JOIN product p on p.id = s.product_id
    ORDER BY s.sales DESC 
    LIMIT 25
    """
    product_data = query_db(query)
    product_data.groupby(['id', 'title', 'image_url', 'details_url',
                          'category', 'prod_material']).sales.sum().reset_index()
    product_data = product_data.sort_values(by='sales', ascending=False)
    data = {'filter': get_filter_data(), 'products': product_data}
    return render_template('dashboard/products.html', data=data)

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    # resize img
    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

@dashboard.route('/dashboard/settings', methods=['GET', 'POST'])
@login_required
def settings():
    data = {'filter': get_filter_data()}

    form = UpdateUserInfo()
    if form.validate_on_submit():
        if form.picture.data:
            picture_name = save_picture(form.picture.data)
            current_user.image = picture_name

        current_user.email = form.email.data
        current_user.first_name = form.first_name.data
        current_user.last_name = form.last_name.data

        db.session.commit()
        flash('Your account has been updated', 'success')
        return redirect(url_for('dashboard.settings'))

    return render_template('dashboard/settings.html', data=data, form=form)


@dashboard.route('/dashboard/calendar')
@login_required
def calendar():
    data = {'filter': get_filter_data()}
    return render_template('dashboard/calendar.html', data=data)


@dashboard.route('/dashboard/products/<id>')
@login_required
def product_details(id):
    prod_query = """SELECT * FROM product p  
        LEFT JOIN stone st ON p.id = st.product_id    
        LEFT JOIN birthstone b ON b.id = st.birthstone_id
        LEFT JOIN shape ON shape.id = st.shape_id
        WHERE p.id = {}""".format(id)

    av_query = f"""SELECT (SELECT count(*) FROM availability WHERE availability = 'In Stock'
    AND product_id =  {id}) / count(*) * 100 AS 'in_stock_percentage',
        (SELECT count(*) FROM availability WHERE availability != 'In Stock'
        AND product_id = {id} ) / count(*) * 100 AS 'oos_percentage'    
        FROM availability
        WHERE product_id = {id}"""
    sales_query = f"""SELECT sum(sales) AS total_sales, (SELECT sum(sales)
        FROM sales WHERE product_id = {id} ) / (SELECT sum(sales) FROM sales) * 100 AS 'share_of_total'
        FROM sales WHERE product_id = {id}"""
    prom_query = """SELECT avg(p.regular_price) AS regular_price, avg(p.promotion_price) AS promotion_price
         FROM promotion p WHERE p.product_id = {}""".format(id)
    rev_query = """SELECT * FROM review
        LEFT JOIN reply ON review.id = reply.review_id
        LEFT JOIN reviewer on review.reviewer_id = reviewer.id
        where product_id = {}""".format(id)

    prod_data = query_db(prod_query)
    if prod_data.shape[0] > 0:
        av_data = query_db(av_query)
        s_data = query_db(sales_query)
        p_data = query_db(prom_query)
        r_data = query_db(rev_query)

        data = {'filter': get_filter_data(), 'product': prod_data, 'availability': av_data, 'sales': s_data,
                'promotions': p_data, 'reviews': r_data}
        return render_template('dashboard/details.html', data=data)
    else:
        return redirect(url_for('not_found'))


@dashboard.route('/404')
def not_found():
    return render_template('404.html')



@dashboard.route('/filter', methods=["GET", "POST"])
@login_required
def filter_data():
    if request.method == "POST":
        imd = ImmutableMultiDict(request.values).to_dict(flat=False)

        brands = imd.get('data[brands][]')
        categories = imd.get('data[categories][]')
        occasions = imd.get('data[occasions][]')
        materials = imd.get('data[materials][]')
        collections = imd.get('data[collections][]')

        print('brands:', brands)
        print('categories:', categories)
        print('occasions:', occasions)
        print('materials:', materials)
        print('collections:', collections)
        return('')

    return redirect(url_for('dashboard.settings'))


@dashboard.route('/bokeh')
def bokeh():
    query = """SELECT p.category, p.brand, p.occasion, p.collection, s.sales, d.month, d.quarter 
                FROM sales s
                LEFT JOIN product p ON s.product_id = p.id
                LEFT JOIN date d on s.date_id = d.id"""
    data = query_db(query);

    genre_list = ['All','Horror', 'Comedy', 'Sci-Fi', 'Action', 'Drama', 'War']

    controls = {
        "reviews": Slider(title="Min # of reviews", value=100, start=100, end=140, step=10),
        "min_year": Slider(title="Start Year", start=2001, end=2005, value=2001, step=1),
        "max_year": Slider(title="End Year", start=2002, end=2006, value=2006, step=1),
        "genre": Select(title="Genre", value="All", options=genre_list)
    }
    controls_array = controls.values()
    source = ColumnDataSource()
    source.data = dict(
        x = [d for d in [4,5,6,7,8,9]],
        y = [d for d in [1,3,4,5,7,8]],
        color = ["#FF9900" for d in [4,1,2,3,12,1]],
        title = [d for d in ['One', 'Two', 'Three', 'four', 'five', 'six']],
        released = [d for d in [2001, 2002, 2003, 2004, 2005, 2006]],
        imdbvotes = [d for d in [100,111,121, 131, 141, 151]],
        genre = [d for d in ['Horror', 'Comedy', 'Sci-Fi', 'Action', 'Drama', 'War']]
    )
    callback = CustomJS(args=dict(source=source, controls=controls), code="""
        if (!window.full_data_save) {
            window.full_data_save = JSON.parse(JSON.stringify(source.data));
        }
        var full_data = window.full_data_save;
        var full_data_length = full_data.x.length;
        var new_data = { x: [], y: [], color: [], title: [], released: [], imdbvotes: [] }
        for (var i = 0; i < full_data_length; i++) {
            if (full_data.imdbvotes[i] === null || full_data.released[i] === null || full_data.genre[i] === null)
                continue;
            if (
                full_data.imdbvotes[i] > controls.reviews.value &&
                Number(full_data.released[i]) >= controls.min_year.value &&
                Number(full_data.released[i]) <= controls.max_year.value &&
                (controls.genre.value === 'All' || full_data.genre[i].split(",").some(ele => ele.trim() === controls.genre.value))
            ) {
                Object.keys(new_data).forEach(key => new_data[key].push(full_data[key][i]));
            }
        }
        
        source.data = new_data;
        source.change.emit();
    """)

    fig = figure(plot_height=600, plot_width=720)

    fig.circle(x="x", y="y", source=source, size=5, color="color", line_color=None)
    fig.xaxis.axis_label = "IMDB Rating"
    fig.yaxis.axis_label = "Rotten Tomatoes Rating"


    for single_control in controls_array:
        single_control.js_on_change('value', callback)

    inputs_column = column(*controls_array, width=320, height=1000)
    layout_row = row([ inputs_column, fig ])

    curdoc().theme = "dark_minimal"
    curdoc().add_root(layout_row)

    script, div = components(layout_row)
    return render_template(
        'bookeh.html',
        plot_script=script,
        plot_div=div,
        js_resources=INLINE.render_js(),
        css_resources=INLINE.render_css(),
    ).encode(encoding='UTF-8')

def create_filters():
    filters = {'category':[], 'brand':[], 'occasion':[], 'collection':[], 'material':[]}
    data = get_filter_data()

    filters['category'] = data.category.unique().tolist()
    filters['brand'] = data.brand.unique().tolist()
    filters['occasion'] = data.occasion.unique().tolist()
    filters['collection'] = data.collection.unique().tolist()
    filters['material'] = data.prod_material.unique().tolist()

    for key in list(filters.keys()):
        filters[key].insert(0, 'All')

    return filters

@dashboard.route('/dashboard/get_data', methods=['POST'])
def get_data():
    query = """SELECT p.id, p.category, p.occasion, p.prod_material, p.brand, p.collection, s.sales, d.month, d.quarter, d.date
                FROM sales s
                LEFT JOIN product p ON s.product_id = p.id
                LEFT JOIN date d on s.date_id = d.id"""
    df = query_db(query);

    df = df.groupby(['month', 'category', 'occasion', 'prod_material', 'brand', 'collection']).sales.sum().reset_index()
    df.month = df.apply(lambda x: datetime.strptime(x.month, "%b").month, axis=1)
    df = df.sort_values(by='month')
    source = ColumnDataSource(df)
    return source

@dashboard.route('/bokeh2')
def bokeh2():
    query = """SELECT p.id, p.category, p.occasion, p.prod_material, p.brand, p.collection, s.sales, d.month, d.quarter, d.date
                FROM sales s
                LEFT JOIN product p ON s.product_id = p.id
                LEFT JOIN date d on s.date_id = d.id"""
    df = query_db(query);

    df = df.groupby(['month', 'category', 'occasion', 'prod_material', 'brand', 'collection']).sales.sum().reset_index()
    df.month = df.apply(lambda x: datetime.strptime(x.month, "%b").month, axis=1)
    df = df.sort_values(by='month')

    print(df[:100])
    filters = create_filters()

    controls = {
        "category": Select(title="Category", value="All", options=filters['category']),
        "occasion": Select(title="Occasion", value="All", options=filters['occasion']),
        "material": Select(title="Material", value="All", options=filters['material']),
        "collection": Select(title="Collection", value="All", options=filters['collection']),
        "brand": Select(title="Brand", value="All", options=filters['brand']),
    }
    controls_array = controls.values()
    source = ColumnDataSource(df)


    callback = CustomJS(args=dict(source=source, controls=controls), code="""
        
        if (!window.full_data_save) {
            window.full_data_save = JSON.parse(JSON.stringify(source.data));
        }
        var full_data = window.full_data_save;
        var full_data_length = full_data.month.length;
        var new_data = { month: [], sales: [], category: [], occasion: [], prod_material: [], collection: [], brand: [] }
        for (var i = 0; i < full_data_length; i++) {
            if (full_data.category[i] === null || full_data.occasion[i] === null || full_data.prod_material[i] === null || full_data.collection[i] === null || full_data.brand[i] === null)
                continue;
            if (
                (controls.category.value === 'All' || full_data.category[i].split(",").some(ele => ele.trim() === controls.category.value)) &&
                (controls.occasion.value === 'All' || full_data.occasion[i].split(",").some(ele => ele.trim() === controls.occasion.value)) &&
                (controls.collection.value === 'All' || full_data.collection[i].split(",").some(ele => ele.trim() === controls.collection.value)) &&
                (controls.material.value === 'All' || full_data.prod_material[i].split(",").some(ele => ele.trim() === controls.material.value)) &&
                (controls.brand.value === 'All' || full_data.brand[i].split(",").some(ele => ele.trim() === controls.brand.value))
            )
            {
                Object.keys(new_data).forEach(key => new_data[key].push(full_data[key][i]));
            }
        }
        source.data = new_data;
        source.change.emit();
    """)


    fig = figure(plot_height=600, plot_width=720)

    fig.line(x='month', y='sales', source=source)
    fig.xaxis.axis_label = "Month"
    fig.yaxis.axis_label = "Sales"


    for single_control in controls_array:
        single_control.js_on_change('value', callback)

    inputs_column = column(*controls_array, width=320, height=1000)
    layout_row = row([ inputs_column, fig ])

    curdoc().theme = "dark_minimal"
    curdoc().add_root(layout_row)

    script, div = components(layout_row)
    return render_template(
        'bookeh.html',
        plot_script=script,
        plot_div=div,
        js_resources=INLINE.render_js(),
        css_resources=INLINE.render_css(),
    ).encode(encoding='UTF-8')

from jinja2 import Template

from bokeh.sampledata.iris import flowers

page = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
  {{ resources }}
</head>
<body>
  <div id="myplot"></div>
  <div id="myplot2"></div>
  <script>
  fetch('/plot')
    .then(function(response) { return response.json(); })
    .then(function(item) { return Bokeh.embed.embed_item(item); })
  </script>
  <script>
  fetch('/plot2')
    .then(function(response) { return response.json(); })
    .then(function(item) { return Bokeh.embed.embed_item(item, "myplot2"); })
  </script>
</body>
""")

colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
colors = [colormap[x] for x in flowers['species']]

def make_plot(x, y):
    p = figure(title = "Iris Morphology", sizing_mode="fixed", width=400, height=400)
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y
    p.circle(flowers[x], flowers[y], color=colors, fill_alpha=0.2, size=10)
    return p
from bokeh.io import curdoc
@dashboard.route('/1')
def root():
    return page.render(resources=CDN.render())

@dashboard.route('/plot')
def plot():
    query = """SELECT p.category, p.brand, p.occasion, p.collection, s.sales, d.month, d.quarter 
                FROM sales s
                LEFT JOIN product p ON s.product_id = p.id
                LEFT JOIN date d on s.date_id = d.id"""
    data = query_db(query);


    p = figure(title = "Iris Morphology", sizing_mode="fixed", width=400, height=400)
    curdoc().theme = "dark_minimal"
    #curdoc().add_root(p)
    p.xaxis.axis_label = 'month'
    p.yaxis.axis_label = 'sales'
    p.toolbar.active_drag=None
    p.circle([12,42,13,54,2,765], [712,131,43,123,512,2], fill_alpha=0.2, size=10)

    return json.dumps(json_item(p, "myplot"))

@dashboard.route('/plot2')
def plot2():
    p = make_plot('sepal_width', 'sepal_length')
    return json.dumps(json_item(p))