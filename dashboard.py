#region Imports
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.datastructures import ImmutableMultiDict
from flask_login import login_required, current_user
from datetime import datetime
from PIL import Image
from bokeh.models import ColumnDataSource, MultiSelect, HoverTool, DataTable, TableColumn
from bokeh.models import FuncTickFormatter, NumberFormatter, NumeralTickFormatter, HTMLTemplateFormatter
from bokeh.models.callbacks import CustomJS
from bokeh.resources import INLINE
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.layouts import row, layout
from bokeh.models.widgets.tables import StringEditor, TextEditor
from bokeh.transform import cumsum, dodge
from numerize import numerize
from math import pi
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
import secrets
import calendar as cal
import store
import os

dashboard = Blueprint('dashboard', __name__)

from app import db, app
from models import UpdateUserInfo

#endregion

#region Database

def connect():
    _db = mysql.connector.connect(user=store.user, password=store.password,
                                  host=store.hostname,
                                  database='newschema')
    return _db

def query_db(query):
    _db = connect()
    data = pd.read_sql(query, _db)
    _db.close()
    return data

def alter_db(query):
    _db = connect()
    _cursor = _db.cursor()
    _cursor.execute(query)
    _cursor.close()
    _db.commit()
    _db.close()

# region Database Query
_colors =['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476', '#a1d99b']
sales_data = query_db("""SELECT p.id as product_id, p.category, p.occasion, p.prod_material, p.brand, p.collection, s.sales, d.month, d.quarter
                    FROM sales s
                    LEFT JOIN product p ON s.product_id = p.id
                    LEFT JOIN date d on s.date_id = d.id LIMIT 10000""")

av_data = query_db("""SELECT p.id as product_id, p.category, p.occasion, p.prod_material, p.brand, p.collection, d.date, d.month, d.quarter, a.availability
                    FROM availability a 
                    LEFT JOIN product p ON a.product_id = p.id
                    LEFT JOIN date d on a.date_id = d.id LIMIT 100000""")
av_data.date = pd.to_datetime(av_data.date, dayfirst=False).dt.strftime("%Y/%m/%d")

pr_data = query_db("""SELECT p.id, p.category, p.occasion, p.prod_material, p.brand, p.collection, d.date,
             d.month, pr.regular_price, pr.promotion_price
             FROM promotion pr 
             LEFT JOIN product p ON pr.product_id = p.id
             LEFT JOIN date d on pr.date_id = d.id             
             LIMIT 100000""")
pr_data['discount'] = pr_data.promotion_price/pr_data.regular_price
pr_data.date = pd.to_datetime(pr_data.date, dayfirst=False).dt.strftime("%Y/%m/%d")

product_data = query_db("""SELECT p.id as product_id, p.title as product_title, 
        p.category, p.occasion, p.collection, p.prod_material, p.brand, d.month,
        p.image_url, d.quarter, sum(s.sales) as sales
        FROM sales s
        LEFT  JOIN product p on p.id = s.product_id
        left join date d on s.date_id = d.id
        group by p.id, d.quarter
        """)

review_data = query_db("""
        SELECT re.product_id, re.rating, re.title AS review_title, 
	    re.review_text, re.review_source, r.location,
        r.gender, r.age_group, re.review_date, rp.reply_text, p.title AS product_title, 
        p.category, p.occasion, p.brand, p.collection, p.prod_material, p.image_url
        FROM review re
        LEFT JOIN reviewer r ON re.reviewer_id = r.id
        LEFT JOIN reply rp ON re.id = rp.review_id
        LEFT JOIN product p ON re.product_id = p.id
        LEFT JOIN date d ON re.review_date = d.date
        """)
review_data.review_date = pd.to_datetime(review_data.review_date, dayfirst=False)
review_data['month'] = review_data.review_date.dt.month
review_data.month = review_data.month.apply(lambda x: cal.month_abbr[x])

text_template = """<span href="#" data-toggle="tooltip" title="<%= value %>" style="font-size: 15px;"><%= value %></span>"""
pr_title_template = """<img src="<%= image_url %>" style="width:50px;height:50px;border:0">
                    <a href="http://127.0.0.1:5000/dashboard/products/<%= product_id %>" target="_blank" style="text-decoration: none; color: black;">
                    <span data-toggle="tooltip" title="<%= value %>" style="font-size: 15px;"><%= value %></span>"""
sales_val_template = """<span href="#" data-toggle="tooltip" title="<%= value %>" style="font-size: 15px;">£ <%= value %></span>"""
# endregion

#endregion

#region Non-Routing Methods

def create_filters(df):
    filters = dict()

    filters['category'] = df.category.unique().tolist()
    filters['brand'] = df.brand.unique().tolist()
    filters['occasion'] = df.occasion.unique().tolist()
    filters['collection'] = df.collection.unique().tolist()
    filters['material'] = df.prod_material.unique().tolist()
    filters['month'] = df.month.unique().tolist()
    return filters

def create_controls(filters):
    controls = {
        "month": MultiSelect(title='Months', value=filters['month'], options=filters['month']),
        "category": MultiSelect(title="Category", value=filters['category'], options=filters['category']),
        "occasion": MultiSelect(title="Occasion", value=filters['occasion'], options=filters['occasion']),
        "material": MultiSelect(title="Material", value=filters['material'], options=filters['material']),
        "collection": MultiSelect(title="Collection", value=filters['collection'], options=filters['collection']),
        "brand": MultiSelect(title="Brand", value=filters['brand'], options=filters['brand']),
    }
    return controls

def get_controls_list(controls):
    controls_array = controls.values()
    return controls_array

def percentage_change(col1,col2):
    pct = ((col2 - col1) / col1) * 100
    if pct > 0:
        return """<span><i class="fas fa-arrow-alt-circle-up text-success"></i></span>"""
    elif pct < 0:
        return """<span><i class="fas fa-arrow-alt-circle-down text-danger"></i></span>"""
    return """<span><i class="fas fa-equals text-warning"></i></span>"""

def human_readable(val):
    return numerize.numerize(val)

def get_cards_data():
    # =============================================================#

    tsales = query_db('SELECT SUM(sales) AS tsales FROM sales')
    tsales_hr = numerize.numerize(tsales.tsales.iloc[0])  # human readable
    # =============================================================#

    av_rating = query_db('SELECT AVG(rating) AS av_rating FROM review')
    av_rating = round(av_rating.av_rating.iloc[0], 1)
    # =============================================================#

    av_availability = query_db(
        """SELECT (SELECT count(*) FROM availability WHERE availability = 'In Stock') / count(*) * 100 AS 'av_availability' FROM availability""")
    av_availability = round(av_availability.av_availability.iloc[0], 1)
    # =============================================================#

    num_reviews = query_db('SELECT COUNT(review_text) AS num_reviews FROM review')
    num_reviews = numerize.numerize(float(num_reviews.num_reviews.iloc[0]))
    # =============================================================#

    return {'sales': tsales_hr, 'rating': av_rating, 'availability': av_availability, 'reviews': num_reviews}

def get_summary_sales():
    df = sales_data.groupby('month').sales.sum().reset_index()
    df.month = df.apply(lambda x: datetime.strptime(x.month, "%b").month, axis=1)
    df = df.sort_values(by='month')
    df.month = df.month.apply(lambda x: cal.month_abbr[x])
    x = df.month
    y = df.sales
    source = ColumnDataSource(data=dict(x=x, y=y))
    plot = figure(sizing_mode='scale_width', plot_height=300, tools='save, reset', x_range=x)
    plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6, line_color='blue')
    plot.xaxis.axis_label = "Month"
    plot.yaxis.axis_label = "Sales"
    plot.toolbar.active_drag = None
    plot.yaxis[0].formatter = FuncTickFormatter(code='''
                                                    if (tick < 1e3){
                                                        var unit = ''
                                                        var num =  (tick).toFixed(2)
                                                      }
                                                      else if (tick < 1e6){
                                                        var unit = 'K'
                                                        var num =  (tick/1e3).toFixed(2)
                                                      }
                                                      else{
                                                        var unit = 'M'
                                                        var num =  (tick/1e6).toFixed(2)
                                                        }
                                                    return `£ ${num} ${unit}`
                                                   '''
                                                )
    plot.add_tools(HoverTool(
        tooltips=[
            ('month', '@{x}'),
            ('sales', '£@{y}'),  # use @{ } for field names with spaces
        ],
        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    ))
    #curdoc().theme = "dark_minimal"
    #curdoc().add_root(plot)
    return plot

def style_plot(plot, title_text):
    plot.background_fill_color = "#24282e"
    plot.background_fill_alpha = 0.5
    plot.border_fill_color = "#24282e"
    plot.outline_line_width = 7
    plot.outline_line_alpha = 0.3
    plot.outline_line_color = "grey"
    plot.xgrid.grid_line_color = "grey"
    plot.ygrid.grid_line_color = "grey"
    plot.title.text = title_text
    plot.title.align = "left"
    plot.title.text_color = "#bd6908"
    plot.title.text_font_size = "25px"
    plot.yaxis.major_label_text_color = "#bd6908"
    plot.xaxis.major_label_text_color = "#bd6908"
    plot.yaxis.axis_label_text_color = "#bd6908"
    plot.xaxis.axis_label_text_color = "#bd6908"
    plot.yaxis.axis_label_text_font_size = "15px"
    plot.xaxis.axis_label_text_font_size = "15px"
    plot.yaxis.major_label_text_font_size = "12px"
    plot.xaxis.major_label_text_font_size = "12px"

    return plot

def filter_data(dataframe, data):
    df = dataframe[(dataframe['category'].isin(data['category[]']) &
                     dataframe['occasion'].isin(data['occasion[]']) &
                     dataframe['collection'].isin(data['collection[]']) &
                     dataframe['prod_material'].isin(data['material[]']) &
                     dataframe['brand'].isin(data['brand[]']) &
                     dataframe['month'].isin(data['month[]']))]

    return df

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

#endregion

# region Refresh Data

#region sales
@dashboard.route("/refresh_sales0", methods=['POST'])
def refresh_sales_0():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(product_data, data)
    qsales = df.groupby(['product_id', 'quarter']).sales.sum().reset_index() \
        .pivot(
        values='sales',
        index='product_id',
        columns='quarter'
    ).reset_index().rename_axis(None, axis=1)
    qsales.rename(columns={'2021-Q1': 'q1', '2021-Q2': 'q2', '2021-Q3': 'q3', '2021-Q4': 'q4'}, inplace=True)

    #region Computation

    qsales['q1_change'] = """<span><i class="fas fa-equals text-warning"></i></span>"""
    try:
        qsales['q2_change'] = qsales.apply(lambda x: percentage_change(x['q1'], x['q2']), axis=1)
    except:
        qsales['q2_change'] = """<span><i class="fas fa-equals text-warning"></i></span>"""

    try:
        qsales['q3_change'] = qsales.apply(lambda x: percentage_change(x['q2'], x['q3']), axis=1)
    except:
        qsales['q3_change'] = """<span><i class="fas fa-equals text-warning"></i></span>"""
    try:
        qsales['q4_change'] = qsales.apply(lambda x: percentage_change(x['q3'], x['q4']), axis=1)
    except:
        qsales['q4_change'] = """<span><i class="fas fa-equals text-warning"></i></span>"""

    qsales['sales']=0
    for c in ['q1', 'q2', 'q3', 'q4']:
        try:
            qsales['sales'] = qsales.apply(lambda x: x['sales']+ x[c], axis=1)
        except:
            continue

    data = qsales.merge(df[['product_id', 'product_title', 'image_url']], how='left', on='product_id')
    data = data.drop_duplicates()
    data = data.sort_values(by='sales', ascending=False)

    try:
        data['q1'] = data.apply(lambda x: human_readable(x['q1']), axis=1)
    except:
        data['q1'] = None
    try:
        data['q2'] = data.apply(lambda x: human_readable(x['q2']), axis=1)
    except:
        data['q2'] = None
    try:
        data['q3'] = data.apply(lambda x: human_readable(x['q3']), axis=1)
    except:
        data['q3'] = None
    try:
        data['q4'] = data.apply(lambda x: human_readable(x['q4']), axis=1)
    except:
        data['q4'] = None
    data['sales'] = data.apply(lambda x: human_readable(x['sales']), axis=1)

    #endregion

    response = jsonify({
        'image_url': list(data.image_url),
        'product_title': list(data.product_title),
        'product_id': list(data.product_id),
        'q1': list(data.q1),
        'q1_change': list(data.q1_change),
        'q2': list(data.q2),
        'q2_change': list(data.q2_change),
        'q3': list(data.q3),
        'q3_change': list(data.q3_change),
        'q4': list(data.q4),
        'q4_change': list(data.q4_change),
        'sales': list(data.sales),
    })
    return response

@dashboard.route("/refresh_sales1", methods=['POST'])
def refresh_sales_1():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(sales_data, data)

    df = df.groupby('month').sales.sum().reset_index()
    df.month = df.apply(lambda x: datetime.strptime(x.month, "%b").month, axis=1)
    df = df.sort_values(by='month')
    df.month = df.month.apply(lambda x: cal.month_abbr[x])

    x = list(df.month)
    y = list(df.sales)
    response = jsonify({'x': x, 'y': y})
    return response

@dashboard.route("/refresh_sales2", methods=['POST'])
def refresh_sales_2():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(sales_data, data)

    df = df.groupby('category').sales.sum().reset_index()
    df = df.sort_values(by='sales')

    x = list(df.sales)
    y = list(df.category)
    response = jsonify({'x': x, 'y': y})
    return response

@dashboard.route("/refresh_sales3", methods=['POST'])
def refresh_sales_3():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(sales_data, data)
    df = df.groupby('category').sales.sum().reset_index()
    ts = df.sales.sum()
    df['percentage'] = df['sales'] / ts * 100
    df['angle'] = df['sales'] / ts * 2 * pi
    df['color'] = _colors[:df.category.shape[0]]
    index = list(df.index)
    sales = list(df.sales)
    category = list(df.category)
    percentage = list(df.percentage)
    angle = list(df.angle)
    color = list(df.color)
    response = jsonify({'sales': sales, 'category': category, 'percentage': percentage, 'angle': angle, 'color': color})

    return response

@dashboard.route("/refresh_sales4", methods=['POST'])
def refresh_sales_4():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(sales_data, data)
    df = df.groupby('quarter').sales.sum().reset_index()
    ts = df.sales.sum()
    df['percentage'] = df['sales'] / ts * 100
    df['angle'] = df['sales'] / ts * 2 * pi
    df['color'] = _colors[:df.quarter.shape[0]]
    index = list(df.index)
    sales = list(df.sales)
    quarter = list(df.quarter)
    percentage = list(df.percentage)
    angle = list(df.angle)
    color = list(df.color)
    response = jsonify({'index':index, 'sales':sales, 'quarter':quarter, 'percentage':percentage, 'angle':angle, 'color':color})
    return response

@dashboard.route("/refresh_sales5", methods=['POST'])
def refresh_sales_5():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(sales_data, data)

    df = df.groupby('occasion').sales.sum().reset_index()

    df = df.sort_values(by='sales')
    x = list(df.occasion)
    y = list(df.sales)
    response = jsonify({'x': x, 'y': y})
    return response
#endregion

#region availability
@dashboard.route("/refresh_av0", methods=['POST'])
def refresh_av_0():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(av_data, data)
    df.date = pd.to_datetime(df.date, dayfirst=False).dt.strftime("%Y/%m/%d")
    tprod = df.product_id.nunique()
    df = df[df.availability == 'In Stock']
    df = df.groupby('date').availability.count().reset_index()
    df.availability = df.availability / tprod
    df = df.sort_values(by='date')
    #df.date = df.date.apply(lambda x: str(x))
    df.rename(columns={'availability': 'In Stock'}, inplace=True)

    x = list(df.date)
    y = list(df['In Stock'])


    response = jsonify({'x':x, 'y':y})
    return response

@dashboard.route("/refresh_av1", methods=['POST'])
def refresh_av_1():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(av_data, data)
    total_instock = df[df['availability'] == 'In Stock']
    y_stock = total_instock.groupby('brand').availability.count().reset_index()
    y_stock.rename(columns={'availability': 'year_stock'}, inplace=True)
    q_stock = total_instock.groupby(['brand', 'quarter']) \
        .availability.count() \
        .reset_index() \
        .pivot(
        values='availability',
        index='brand',
        columns='quarter')
    q_stock.rename(columns={'2021-Q1': 'q1_stock', '2021-Q2': 'q2_stock', '2021-Q3': 'q3_stock', '2021-Q4': 'q4_stock'},
                   inplace=True)

    total_y = df.groupby('brand').availability.count().reset_index()
    total_y.rename(columns={'availability': 'year_total'}, inplace=True)
    total_q = df.groupby(['brand', 'quarter']) \
        .availability.count() \
        .reset_index() \
        .pivot(
        values='availability',
        index='brand',
        columns='quarter'
    ).reset_index()
    total_q.rename(
        columns={'2021-Q1': 'tq1_stock', '2021-Q2': 'tq2_stock', '2021-Q3': 'tq3_stock', '2021-Q4': 'tq4_stock'},
        inplace=True)
    avg_y = y_stock.merge(total_y, how='left', on='brand')
    avg_y['avg_y'] = avg_y['year_stock'] / avg_y['year_total']  ## average yearly stock by brand

    avg_q = q_stock.merge(total_q, how='left', on='brand')
    try:
        avg_q['avg_q1'] = avg_q['q1_stock'] / avg_q['tq1_stock']
    except:
        avg_q['avg_q1'] = 0
    try:
        avg_q['avg_q2'] = avg_q['q2_stock'] / avg_q['tq2_stock']
    except:
        avg_q['avg_q2'] = 0
    try:
        avg_q['avg_q3'] = avg_q['q3_stock'] / avg_q['tq3_stock']
    except:
        avg_q['avg_q3'] = 0
    try:
        avg_q['avg_q4'] = avg_q['q4_stock'] / avg_q['tq4_stock']
    except:
        avg_q['avg_q4'] = 0

    data = avg_y[['brand', 'avg_y']].merge(
        avg_q[['brand', 'avg_q1', 'avg_q2', 'avg_q3', 'avg_q4']],
        how='left',
        on='brand'
    )
    brand = list(data.brand)
    year = list(data.avg_y)
    q1 = list(data.avg_q1)
    q2 = list(data.avg_q2)
    q3 = list(data.avg_q3)
    q4 = list(data.avg_q4)

    response = jsonify({'brand':brand, 'avg_y':year, 'avg_q1':q1, 'avg_q2':q2, 'avg_q3':q3, 'avg_q4':q4 })
    return response

@dashboard.route("/refresh_av2", methods=['POST'])
def refresh_av_2():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(av_data, data)
    in_stock = df[df.availability == 'In Stock'].groupby('category').availability.count().reset_index()
    total = df.groupby('category').availability.count().reset_index()
    total.rename(columns={'availability': 'total'}, inplace=True)
    in_stock = in_stock.merge(total, how='left', on='category')
    in_stock['Stock %'] = in_stock['availability'] / in_stock['total']
    av_plot2_df = in_stock[['category', 'Stock %']]
    x = list(av_plot2_df['Stock %'])
    y = list(av_plot2_df['category'])

    response=jsonify({'x':x, 'y':y})
    return response

@dashboard.route("/refresh_av3", methods=['POST'])
def refresh_av_3():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(av_data, data)
    in_stock = df[df.availability == 'In Stock'].groupby('occasion').availability.count().reset_index()
    total = df.groupby('occasion').availability.count().reset_index()
    total.rename(columns={'availability': 'total'}, inplace=True)
    in_stock = in_stock.merge(total, how='left', on='occasion')
    in_stock['Stock %'] = in_stock['availability'] / in_stock['total']
    av_plot2_df = in_stock[['occasion', 'Stock %']]
    y = list(av_plot2_df['Stock %'])
    x = list(av_plot2_df['occasion'])

    response=jsonify({'x':x, 'y':y})
    return response

#endregion

#region promotion
@dashboard.route("/refresh_pr1", methods=['POST'])
def refresh_pr_1():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(pr_data, data)

    df = df[df.promotion_price > 0].groupby('month').promotion_price.count().reset_index()
    df.month = df.apply(lambda x: datetime.strptime(x.month, "%b").month, axis=1)
    df = df.sort_values(by='month')
    df.month = df.month.apply(lambda x: cal.month_abbr[x])

    x = list(df.month)
    y = list(df.promotion_price)
    response = jsonify({'x': x, 'y': y})
    return response

@dashboard.route("/refresh_pr2", methods=['POST'])
def refresh_pr_2():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(pr_data, data)

    promo_price = df.groupby('category').promotion_price.mean().reset_index()
    promo_price.rename(columns={'promotion_price': 'avg_promotion_price'}, inplace=True)
    promo_count = df[df.promotion_price > 0].groupby('category').promotion_price.count().reset_index()
    promo_count.rename(columns={'promotion_price': 'total_promotions'}, inplace=True)
    reg_price = df.groupby('category').regular_price.mean().reset_index()
    reg_price.rename(columns={'regular_price': 'avg_regular_price'}, inplace=True)
    data = promo_price.merge(promo_count, how='left', on='category')
    data = data.merge(reg_price, how='left', on='category')
    data['avg_discount'] = (data['avg_regular_price'] - data['avg_promotion_price'])/data['avg_regular_price']
    data['avg_promotion_price'] = data.apply(lambda x: human_readable(x['avg_promotion_price']), axis=1)
    data['avg_regular_price'] = data.apply(lambda x: human_readable(x['avg_regular_price']), axis=1)
    category = data.category.tolist()
    total_promotions=data.total_promotions.tolist()
    avg_regular_price=data.avg_regular_price.tolist()
    avg_promotion_price=data.avg_promotion_price.tolist()
    avg_discount=data.avg_discount.tolist()

    response = jsonify({'category': category, 'total_promotions': total_promotions,
                        'avg_regular_price':avg_regular_price, 'avg_promotion_price':avg_promotion_price,
                        'avg_discount':avg_discount})
    return response

#endregion

#region reviews

@dashboard.route("/refresh_re1", methods=['POST'])
def refresh_reviews_1():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(review_data, data)

    t_rev = df.shape[0]
    df = df.groupby('gender').rating.count().reset_index()
    df.rating = df.rating / t_rev

    x = list(df.gender)
    y = list(df.rating)
    response = jsonify({'x': x, 'y': y})
    return response

@dashboard.route("/refresh_re2", methods=['POST'])
def refresh_reviews_2():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(review_data, data)
    t_rev = df.shape[0]
    df = df.groupby('review_source').rating.count().reset_index()
    df.rating = df.rating / t_rev

    x = list(df.review_source)
    y = list(df.rating)
    response = jsonify({'x': x, 'y': y})
    return response

@dashboard.route("/refresh_re3", methods=['POST'])
def refresh_reviews_3():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(review_data, data)
    t_rev = df.shape[0]
    df.loc[df.rating <= 3, 'review_score'] = 'negative'
    df.loc[df.rating > 3, 'review_score'] = 'positive'
    df = df.groupby('review_score').rating.count().reset_index()
    df.rating = df.rating / t_rev

    x = list(df.review_score)
    y = list(df.rating)
    response = jsonify({'x': x, 'y': y})
    return response

@dashboard.route("/refresh_re4", methods=['POST'])
def refresh_reviews_4():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(review_data, data)
    t_rev = df.shape[0]
    #df['rating_hr'] = df.apply(lambda x: str(x['rating']) + ' stars', axis=1)
    df['rating_hr'] = [str(ele) + ' starts' for ele in df['rating']]
    df = df.groupby('rating_hr').review_source.count().reset_index()
    df.review_source = df.review_source / t_rev

    x = list(df.rating_hr)
    y = list(df.review_source)
    response = jsonify({'x': x, 'y': y})
    return response

@dashboard.route("/refresh_re5", methods=['POST'])
def refresh_reviews_5():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(review_data, data)
    t_rev = df.shape[0]
    df = df.groupby('age_group').rating.count().reset_index()
    df.rating = df.rating / t_rev

    x = list(df.age_group)
    y = list(df.rating)
    response = jsonify({'x': x, 'y': y})
    return response

@dashboard.route("/refresh_re6", methods=['POST'])
def refresh_reviews_6():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_data(review_data, data)
    df = df[['product_id', 'review_date', 'image_url', 'review_title', 'review_text', 'product_title', 'rating']].copy()
    df.review_date = pd.to_datetime(df.review_date).dt.strftime('%d-%m-%Y')
    df.fillna('', inplace=True)
    df.loc[
        df.image_url == '', 'image_url'] = 'https://i1.wp.com/fremontgurdwara.org/wp-content/uploads/2020/06/no-image-icon-2.png'
    df.loc[df.product_title == '', 'product_title'] = 'Product Unknown'
    df.loc[df.product_id == '', 'product_id'] = 0

    response=jsonify({
        'review_date' : list(df.review_date),
        'image_url' : list(df.image_url),
        'review_title' : list(df.review_title),
        'review_text' : list(df.review_text),
        'product_title' : list(df.product_title),
        'product_id' : list(df.product_id),
        'rating' : list(df.rating),
    })

    return response

#endregion

# endregion

#region Pages

@dashboard.route('/dashboard/summary')
@login_required
def summary():
    script_summary_sales, div_summary_sales = components(get_summary_sales())
    script_summary_sales2, div_summary_sales2 = components(get_summary_sales())
    card_data = get_cards_data()


    return render_template(
        'dashboard/dashboard.html',
        card_data = card_data,
        ch1_plot_script=script_summary_sales,
        ch1_plot_script2=script_summary_sales2,
        ch1_plot_div=div_summary_sales,
        ch1_plot_div2=div_summary_sales2,
        ch1_js_resources=INLINE.render_js(),
        ch1_css_resources=INLINE.render_css(),
    ).encode(encoding='UTF-8')

@dashboard.route('/dashboard/sales')
@login_required
def sales():
    #region Filters
    filters = create_filters(sales_data) # Create the filters
    controls = create_controls(filters) # Create the Controls for filters
    controls_array = get_controls_list(controls)
    #endregion

    # region DataTable
    # region data
    qsales = product_data.groupby(['product_id', 'quarter']).sales.sum().reset_index() \
        .pivot(
        values='sales',
        index='product_id',
        columns='quarter'
    ).reset_index().rename_axis(None, axis=1)
    qsales.rename(columns={'2021-Q1': 'q1', '2021-Q2': 'q2', '2021-Q3': 'q3', '2021-Q4': 'q4'}, inplace=True)

    qsales['q1_change'] = """<span><i class="fas fa-equals text-warning"></i></span>"""
    qsales['q2_change'] = qsales.apply(lambda x: percentage_change(x['q1'], x['q2']), axis=1)
    qsales['q3_change'] = qsales.apply(lambda x: percentage_change(x['q2'], x['q3']), axis=1)
    qsales['q4_change'] = qsales.apply(lambda x: percentage_change(x['q3'], x['q4']), axis=1)
    qsales['sales'] = qsales.apply(lambda x: x['q1'] + x['q2'] + x['q3'] + x['q4'], axis=1)

    data = qsales.merge(product_data[['product_id', 'product_title', 'image_url']], how='left', on='product_id')
    data = data.drop_duplicates()
    data = data.sort_values(by='sales', ascending=False)
    data['q1'] = data.apply(lambda x: human_readable(x['q1']), axis=1)
    data['q2'] = data.apply(lambda x: human_readable(x['q2']), axis=1)
    data['q3'] = data.apply(lambda x: human_readable(x['q3']), axis=1)
    data['q4'] = data.apply(lambda x: human_readable(x['q4']), axis=1)
    data['sales'] = data.apply(lambda x: human_readable(x['sales']), axis=1)
    #endregion

    dt_source2 = ColumnDataSource(data)
    columns = [
        TableColumn(field='product_title', title="Product", formatter=HTMLTemplateFormatter(template=pr_title_template)),
        TableColumn(field='q1', title="Q1", formatter=HTMLTemplateFormatter(template= sales_val_template + " <%= q1_change %>")),
        TableColumn(field='q2', title="Q2", formatter=HTMLTemplateFormatter(template= sales_val_template + " <%= q2_change %>")),
        TableColumn(field='q3', title="Q3", formatter=HTMLTemplateFormatter(template= sales_val_template + " <%= q3_change %>")),
        TableColumn(field='q4', title="Q4", formatter=HTMLTemplateFormatter(template= sales_val_template + " <%= q4_change %>")),
        TableColumn(field='sales', title="Total Sales", formatter=HTMLTemplateFormatter(template= sales_val_template)),
    ]
    data_table2 = DataTable(source=dt_source2, columns=columns, height=500, row_height=50, index_position=None,
                           sizing_mode='stretch_width', background=None, css_classes=['bokeh-table'])

    dt_callback = CustomJS(args=dict(source=dt_source2, controls=controls), code="""
            var selected_value = new Object();

            for (let key of Object.keys(controls)) {
                var val = controls[key].value;
                // console.log(key, val);
                selected_value[key] = val;
            }
            //alert(Object.keys(controls));
            var plot_data = source.data;

            jQuery.ajax({
                type: 'POST',
                url: '/refresh_sales0',
                data: selected_value,
                dataType: 'json',
                success: function (json_from_server) {
                    plot_data.product_title = json_from_server.product_title;
                    plot_data.image_url = json_from_server.image_url; 
                    plot_data.q1 = json_from_server.q1; 
                    plot_data.q1_change = json_from_server.q1_change; 
                    plot_data.q2 = json_from_server.q2; 
                    plot_data.q2_change = json_from_server.q2_change; 
                    plot_data.q3 = json_from_server.q3; 
                    plot_data.q3_change = json_from_server.q3_change; 
                    plot_data.q4 = json_from_server.q4; 
                    plot_data.q4_change = json_from_server.q4_change; 
                    plot_data.sales = json_from_server.sales; 
                    source.change.emit();
                },
                error: function() {
                    alert("Oh no, something went wrong. Search for an error " +
                          "message in Flask log and browser developer tools.");
                }
            });
            """)
    # endregion

    #region Plot1
    plot1_df = sales_data.groupby('month').sales.sum().reset_index()
    plot1_df.month = plot1_df.apply(lambda x: datetime.strptime(x.month, "%b").month, axis=1)
    plot1_df = plot1_df.sort_values(by='month')
    plot1_df.month = plot1_df.month.apply(lambda x: cal.month_abbr[x])
    plot1_x = plot1_df.month
    plot1_y = plot1_df.sales
    plot1_source = ColumnDataSource(data=dict(x=plot1_x, y=plot1_y))
    plot1 = figure(x_range=plot1_x, plot_height=400, tools='save')
    plot1 = style_plot(plot1, 'Sales by Month')
    plot1.line('x', 'y', source=plot1_source, line_width=3, line_alpha=0.6)
    plot1.xaxis.axis_label = "Month"
    plot1.yaxis.axis_label = "Sales"
    plot1.xaxis.major_label_orientation = pi/4
    plot1.yaxis[0].formatter = FuncTickFormatter(code='''
                                                    if (tick < 1e3){
                                                        var unit = ''
                                                        var num =  (tick).toFixed(2)
                                                      }
                                                      else if (tick < 1e6){
                                                        var unit = 'k'
                                                        var num =  (tick/1e3).toFixed(2)
                                                      }
                                                      else{
                                                        var unit = 'm'
                                                        var num =  (tick/1e6).toFixed(2)
                                                        }
                                                    return `£ ${num} ${unit}`
                                                   ''')
    plot1.add_tools(HoverTool(
        tooltips=[
            ('Month', '@{x}'),
            ('Sales', '£@y{0.00 a}'),
        ],
        mode='vline'
    ))
    plot1_callback = CustomJS(args=dict(source=plot1_source, controls=controls), code="""
        var selected_value = new Object();

        for (let key of Object.keys(controls)) {
            var val = controls[key].value;
            // console.log(key, val);
            selected_value[key] = val;
        }
        //alert(Object.keys(controls));
        var plot_data = source.data;

        jQuery.ajax({
            type: 'POST',
            url: '/refresh_sales1',
            data: selected_value,
            dataType: 'json',
            success: function (json_from_server) {
                plot_data.y = json_from_server.y;
                plot_data.x = json_from_server.x; 
                // source.data = json_from_server.data;
                source.change.emit();
            },
            error: function() {
                alert("Oh no, something went wrong. Search for an error " +
                      "message in Flask log and browser developer tools.");
            }
        });
        """)
    #endregion

    #region Plot2
    plot2_df = sales_data.groupby('category').sales.sum().reset_index()
    plot2_df = plot2_df.sort_values(by='sales')
    plot2_x = plot2_df.sales
    plot2_y = plot2_df.category
    plot2_source = ColumnDataSource(data=dict(x=plot2_x, y=plot2_y))
    plot2 = figure(y_range=plot2_y, plot_height=400, tools='save')
    plot2 = style_plot(plot2, 'Category Sales')
    plot2.hbar(y=dodge('y', -0.25, range=plot2.y_range), right='x', height=0.5, source=plot2_source,
               color="#c9d9d3")
    plot2.xaxis.axis_label = "Sales"
    plot2.yaxis.axis_label = "Category"
    plot2.xaxis.major_label_orientation = pi/4
    plot2.xaxis[0].formatter = FuncTickFormatter(code='''
                                                    if (tick < 1e3){
                                                        var unit = ''
                                                        var num =  (tick).toFixed(2)
                                                      }
                                                      else if (tick < 1e6){
                                                        var unit = 'k'
                                                        var num =  (tick/1e3).toFixed(2)
                                                      }
                                                      else{
                                                        var unit = 'm'
                                                        var num =  (tick/1e6).toFixed(2)
                                                        }
                                                    return `£ ${num} ${unit}`
                                                   '''
                                                 )
    plot2.ygrid.grid_line_color = None
    plot2.add_tools(HoverTool(
        tooltips=[
            ('Category', '@{y}'),
            ('Sales', '£@x{0.00 a}'),
        ],
        mode='hline'
    ))
    plot2_callback = CustomJS(args=dict(source=plot2_source, controls=controls), code="""
            var selected_value = new Object();

            for (let key of Object.keys(controls)) {
                var val = controls[key].value;
                // console.log(key, val);
                selected_value[key] = val;
            }
            //alert(Object.keys(controls));
            var plot_data = source.data;

            jQuery.ajax({
                type: 'POST',
                url: '/refresh_sales2',
                data: selected_value,
                dataType: 'json',
                success: function (json_from_server) {
                    plot_data.y = json_from_server.y;
                    plot_data.x = json_from_server.x; 
                    // source.data = json_from_server.data;
                    source.change.emit();
                },
                error: function() {
                    alert("Oh no, something went wrong. Search for an error " +
                          "message in Flask log and browser developer tools.");
                }
            });
            """)
    #endregion

    #region Plot3
    plot3_df = sales_data.groupby('category').sales.sum().reset_index()
    ts = plot3_df.sales.sum()
    plot3_df['percentage'] = plot3_df['sales'] / ts * 100
    plot3_df['angle'] = plot3_df['sales'] / ts * 2 * pi
    plot3_df['color'] = _colors[:plot3_df.category.shape[0]]
    plot3_source = ColumnDataSource(plot3_df)

    plot3 = figure(plot_height=400, tools='save')
    plot3 = style_plot(plot3, 'Category Share')
    plot3.xgrid.grid_line_color = None
    plot3.ygrid.grid_line_color = None
    plot3.wedge(x=0, y=0, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='category', source=plot3_source)
    plot3.legend.location = 'top_left'
    plot3.legend.click_policy = 'mute'
    plot3.add_tools(HoverTool(
        tooltips=[
            ('Category', '@{category}'),
            ('Share', '@percentage{0.2f}%'),
            ('Sales', '£@sales{0.00 a}'),
        ]
    ))
    plot3_callback = CustomJS(args=dict(source=plot3_source, controls=controls), code="""
                    var selected_value = new Object();
                    for (let key of Object.keys(controls)) {
                        var val = controls[key].value;
                        // console.log(key, val);
                        selected_value[key] = val;
                    }
                    //alert(Object.keys(controls));
                    var plot_data = source.data;

                    jQuery.ajax({
                        type: 'POST',
                        url: '/refresh_sales3',
                        data: selected_value,
                        dataType: 'json',
                        success: function (json_from_server) {                        
                            plot_data.angle = json_from_server.angle;
                            plot_data.category = json_from_server.category;
                            plot_data.color = json_from_server.color;
                            plot_data.percentage = json_from_server.percentage; 
                            plot_data.sales = json_from_server.sales; 
                            
                            source.change.emit();
                            source.change.emit();
                            source.change.emit();
                            source.change.emit();
                            source.change.emit();
                            
                            //alert(plot_data.category);    
                        },
                        error: function() {
                            alert("Oh no, something went wrong. Search for an error " +
                                  "message in Flask log and browser developer tools.");
                        }
                    });
                    """)


    #endregion

    #region Plot4
    plot4_df = sales_data.groupby('quarter').sales.sum().reset_index()
    qts = plot4_df.sales.sum()
    plot4_df['percentage'] = plot4_df['sales'] / qts * 100
    plot4_df['angle'] = plot4_df['sales'] / qts * 2 * pi
    plot4_df['color'] = _colors[:plot4_df.quarter.shape[0]]
    plot4_source = ColumnDataSource(plot4_df)

    plot4 = figure(plot_height=400, tools='save, reset')
    plot4 = style_plot(plot4, 'Quarter Sales')
    plot4.xgrid.grid_line_color = None
    plot4.ygrid.grid_line_color = None
    plot4.wedge(x=0, y=0, radius=0.4,
                start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                line_color="white", fill_color='color', legend_field='quarter', source=plot4_source)
    plot4.legend.location='top_left'
    plot4.legend.click_policy = 'mute'
    plot4.add_tools(HoverTool(
        tooltips=[
            ('Quarter', '@{quarter}'),
            ('Share', '@percentage{0.2f}%'),
            ('Sales', '£@sales{0.00 a}'),
        ]
    ))
    plot4_callback = CustomJS(args=dict(source=plot4_source, controls=controls), code="""
                        var selected_value = new Object();
                        for (let key of Object.keys(controls)) {
                            var val = controls[key].value;
                            // console.log(key, val);
                            selected_value[key] = val;
                        }
                        //alert(Object.keys(controls));
                        var plot_data = source.data;

                        jQuery.ajax({
                            type: 'POST',
                            url: '/refresh_sales4',
                            data: selected_value,
                            dataType: 'json',
                            success: function (json_from_server) {                                                    
                                plot_data.angle = json_from_server.angle;
                                plot_data.quarter = json_from_server.quarter;
                                plot_data.color = json_from_server.color;
                                plot_data.index = json_from_server.index;
                                plot_data.percentage = json_from_server.percentage; 
                                plot_data.sales = json_from_server.sales;
                                source.change.emit();
                                //alert(Object.keys(plot_data));        
                            },
                            error: function() {
                                alert("Oh no, something went wrong. Search for an error " +
                                      "message in Flask log and browser developer tools.");
                            }
                        });
                        """)
    #endregion

    #region Plot5
    plot5_df = sales_data.groupby('occasion').sales.sum().reset_index()
    plot5_df = plot5_df.sort_values(by='sales', ascending=False)
    plot5_x = plot5_df.occasion
    plot5_y = plot5_df.sales
    plot5_source = ColumnDataSource(data=dict(x=plot5_x, y=plot5_y))
    plot5 = figure(x_range=plot5_x, plot_height=400, tools='save')
    plot5 = style_plot(plot5, 'Sales by Occasion')
    plot5.vbar(x='x',
               top='y',
               width=0.9,
               source=plot5_source,
               color="#c9d9d3")
    plot5.xgrid.grid_line_color = None
    plot5.y_range.start = 0
    plot5.xaxis.axis_label = "Occasion"
    plot5.yaxis.axis_label = "Sales"
    plot5.xaxis.major_label_orientation = pi/ 4
    plot5.yaxis[0].formatter = FuncTickFormatter(code='''
                                                        if (tick < 1e3){
                                                            var unit = ''
                                                            var num =  (tick).toFixed(2)
                                                          }
                                                          else if (tick < 1e6){
                                                            var unit = 'k'
                                                            var num =  (tick/1e3).toFixed(2)
                                                          }
                                                          else{
                                                            var unit = 'm'
                                                            var num =  (tick/1e6).toFixed(2)
                                                            }
                                                        return `£ ${num} ${unit}`
                                                       '''
                                                 )
    plot5.add_tools(HoverTool(
        tooltips=[
            ('Occasion', '@{x}'),
            ('Sales', '£@y{0.00 a}'),
        ],
        mode='vline'
    ))
    plot5_callback = CustomJS(args=dict(source=plot5_source, controls=controls), code="""
                var selected_value = new Object();

                for (let key of Object.keys(controls)) {
                    var val = controls[key].value;
                    // console.log(key, val);
                    selected_value[key] = val;
                }
                //alert(Object.keys(controls));
                var plot_data = source.data;

                jQuery.ajax({
                    type: 'POST',
                    url: '/refresh_sales5',
                    data: selected_value,
                    dataType: 'json',
                    success: function (json_from_server) {
                        plot_data.y = json_from_server.y;
                        plot_data.x = json_from_server.x; 
                        // source.data = json_from_server.data;
                        source.change.emit();
                    },
                    error: function() {
                        alert("Oh no, something went wrong. Search for an error " +
                              "message in Flask log and browser developer tools.");
                    }
                });
                """)
    #endregion

    #region Callbacks

    for single_control in controls_array:
        single_control.js_on_change('value', plot1_callback)
        single_control.js_on_change('value', plot2_callback)
        single_control.js_on_change('value', plot3_callback)
        single_control.js_on_change('value', plot4_callback)
        single_control.js_on_change('value', plot5_callback)
        single_control.js_on_change('value', dt_callback)

    #endregion

    #region Layout
    filters_row = row(*controls_array, background="#24282e", margin=(10,0,10,0))

    _layout = layout(children=[
            [filters_row],
            [plot2, plot5],
            [plot1],
            #[data_table],
            [plot3, plot4],
            [data_table2]
        ],
    sizing_mode='stretch_width')
    script, div = components(_layout)

    #endregion

    return render_template('dashboard/sales.html',
                           script=script,
                           div=div,
                           js_resources = INLINE.render_js(),
                           css_resources=INLINE.render_css()
                           ).encode(encoding='UTF-8')

@dashboard.route('/dashboard/availability')
@login_required
def availability():

    # region Filters
    filters = create_filters(av_data)  # Create the filters
    controls = create_controls(filters)  # Create the Controls for filters
    controls_array = get_controls_list(controls)
    # endregion

    # region Plot1
    av_plot1_df = av_data[av_data.availability == 'In Stock']
    tprod = av_plot1_df.product_id.nunique()
    av_plot1_df = av_plot1_df.groupby('date').availability.count().reset_index()
    av_plot1_df.availability = av_plot1_df.availability/tprod
    av_plot1_df = av_plot1_df.sort_values(by='date')
    av_plot1_df.rename(columns={'availability': 'In Stock'}, inplace=True)
    #av_plot1['month'] = pd.to_datetime(av_plot1.date).dt.strftime('%B').tolist()
    #av_plot1_df.date = av_plot1_df.date.apply(lambda x: str(x))

    av_plot1_x = av_plot1_df.date
    av_plot1_y = av_plot1_df['In Stock']
    av_plot1_source = ColumnDataSource(data=dict(x=av_plot1_x, y=av_plot1_y))
    av_plot1 = figure(x_range=av_plot1_x, plot_height=400, tools='save')
    av_plot1.y_range.start=0
    av_plot1.y_range.end=1.05
    av_plot1 = style_plot(av_plot1, 'Availability')
    av_plot1.line('x', 'y', source=av_plot1_source, line_width=3, line_alpha=0.6)
    av_plot1.xaxis.axis_label = "Date"
    av_plot1.yaxis.axis_label = "Availability"
    av_plot1.xaxis.major_label_orientation = pi/4
    av_plot1.add_tools(HoverTool(
        tooltips=[
            ('Date', '@{x}'),
            ('Availability', '@y{0.0%}'),
        ],
        mode='vline'
    ))
    av_plot1.yaxis.formatter = NumeralTickFormatter(format='0%')
    #plot1.xaxis.ticker = [1,2,3,4,5,6]

    av_plot1_callback = CustomJS(args=dict(source=av_plot1_source, controls=controls), code="""
            var selected_value = new Object();

            for (let key of Object.keys(controls)) {
                var val = controls[key].value;
                // console.log(key, val);
                selected_value[key] = val;
            }
            //alert(Object.keys(controls));
            var plot_data = source.data;

            jQuery.ajax({
                type: 'POST',
                url: '/refresh_av0',
                data: selected_value,
                dataType: 'json',
                success: function (json_from_server) {
                    //alert(Object.keys(json_from_server));
                    //alert(Object.values(json_from_server));
                    plot_data.y = json_from_server.y;
                    plot_data.x = json_from_server.x; 
                    source.change.emit();
                },
                error: function() {
                    
                }
            });
            """)
    # endregion

    # region DataTable1
    total_instock = av_data[av_data['availability']=='In Stock']
    y_stock = total_instock.groupby('brand').availability.count().reset_index()
    y_stock.rename(columns={'availability': 'year_stock'}, inplace=True)
    q_stock = total_instock.groupby(['brand', 'quarter']) \
        .availability.count() \
        .reset_index() \
        .pivot(
        values='availability',
        index='brand',
        columns='quarter')
    q_stock.rename(columns={'2021-Q1':'q1_stock', '2021-Q2':'q2_stock', '2021-Q3':'q3_stock', '2021-Q4':'q4_stock'}, inplace=True)

    total_y = av_data.groupby('brand').availability.count().reset_index()
    total_y.rename(columns={'availability': 'year_total'}, inplace=True)
    total_q = av_data.groupby(['brand', 'quarter']) \
        .availability.count() \
        .reset_index() \
        .pivot(
        values='availability',
        index='brand',
        columns='quarter'
    ).reset_index()
    total_q.rename(columns={'2021-Q1': 'tq1_stock', '2021-Q2': 'tq2_stock', '2021-Q3': 'tq3_stock', '2021-Q4': 'tq4_stock'}, inplace=True)
    avg_y = y_stock.merge(total_y, how='left', on='brand')
    avg_y['avg_y'] = avg_y['year_stock']/avg_y['year_total'] ## average yearly stock by brand

    avg_q = q_stock.merge(total_q, how='left', on='brand')
    avg_q['avg_q1'] = avg_q['q1_stock']/avg_q['tq1_stock']
    avg_q['avg_q2'] = avg_q['q2_stock']/avg_q['tq2_stock']
    avg_q['avg_q3'] = avg_q['q3_stock']/avg_q['tq3_stock']
    avg_q['avg_q4'] = avg_q['q4_stock']/avg_q['tq4_stock']

    data = avg_y[['brand', 'avg_y']].merge(
        avg_q[['brand', 'avg_q1', 'avg_q2', 'avg_q3', 'avg_q4']],
        how='left',
        on='brand'
    )
    data = data.sort_values(by='avg_y', ascending=False)

    dt_source = ColumnDataSource(data)
    columns = [
        TableColumn(field='brand', title="Brand"),
        TableColumn(field='avg_q1', title="Q1", formatter=NumberFormatter(format='0.00%')),
        TableColumn(field='avg_q2', title="Q2", formatter=NumberFormatter(format='0.00%')),
        TableColumn(field='avg_q3', title="Q3", formatter=NumberFormatter(format='0.00%')),
        TableColumn(field='avg_q4', title="Q4", formatter=NumberFormatter(format='0.00%')),
        TableColumn(field='avg_y', title="AVG Year", formatter=NumberFormatter(format='0.00%')),
    ]
    data_table1 = DataTable(source=dt_source, columns=columns, height=500, index_position=None,
                           auto_edit=True, sizing_mode='stretch_width', background=None, css_classes=['bokeh-table']
                           )

    dt1_callback = CustomJS(args=dict(source=dt_source, controls=controls), code="""
               var selected_value = new Object();

               for (let key of Object.keys(controls)) {
                   var val = controls[key].value;
                   // console.log(key, val);
                   selected_value[key] = val;
               }
               //alert(Object.keys(controls));
               var plot_data = source.data;

               jQuery.ajax({
                   type: 'POST',
                   url: '/refresh_av1',
                   data: selected_value,
                   dataType: 'json',
                   success: function (json_from_server) {
                       plot_data.brand = json_from_server.brand;
                       plot_data.avg_y = json_from_server.avg_y; 
                       plot_data.avg_q1 = json_from_server.avg_q1; 
                       plot_data.avg_q2 = json_from_server.avg_q2; 
                       plot_data.avg_q3 = json_from_server.avg_q3; 
                       plot_data.avg_q4 = json_from_server.avg_q4; 
                       
                       source.change.emit();
                   },
                   error: function() {
                   }
               });
               """)
    # endregion

    #region Plot2
    in_stock = av_data[av_data.availability=='In Stock'].groupby('category').availability.count().reset_index()
    total = av_data.groupby('category').availability.count().reset_index()
    total.rename(columns={'availability':'total'}, inplace=True)
    in_stock = in_stock.merge(total, how='left', on='category')
    in_stock['Stock %'] = in_stock['availability']/in_stock['total']
    av_plot2_df = in_stock[['category', 'Stock %']]
    av_plot2_x = av_plot2_df['Stock %']
    av_plot2_y = av_plot2_df['category']
    av_plot2_source = ColumnDataSource(data=dict(x=av_plot2_x, y=av_plot2_y))
    av_plot2 = figure(y_range=av_plot2_y, plot_height=400, tools='save')
    av_plot2 = style_plot(av_plot2, 'Category Availability')
    av_plot2.hbar(y=dodge('y', -0.25, range=av_plot2.y_range), right='x', height=0.5, source=av_plot2_source,
               color="#c9d9d3")
    av_plot2.xaxis.axis_label = "Stock %"
    av_plot2.yaxis.axis_label = "Category"
    av_plot2.xaxis.major_label_orientation = pi / 4
    av_plot2.ygrid.grid_line_color = None
    av_plot2.xaxis.formatter = NumeralTickFormatter(format='0%')
    av_plot2.x_range.start = 0
    av_plot2.x_range.end = 1
    av_plot2.add_tools(HoverTool(
        tooltips=[
            ('Category', '@{y}'),
            ('Stock %', '@x{0.0%}'),
        ],
        mode='hline'
    ))
    av_plot2_callback = CustomJS(args=dict(source=av_plot2_source, controls=controls), code="""
                var selected_value = new Object();

                for (let key of Object.keys(controls)) {
                    var val = controls[key].value;
                    // console.log(key, val);
                    selected_value[key] = val;
                }
                //alert(Object.keys(controls));
                var plot_data = source.data;

                jQuery.ajax({
                    type: 'POST',
                    url: '/refresh_av2',
                    data: selected_value,
                    dataType: 'json',
                    success: function (json_from_server) {
                        plot_data.y = json_from_server.y;
                        plot_data.x = json_from_server.x; 
                        source.change.emit();
                    },
                    error: function() {
                        alert("Oh no, something went wrong. Search for an error " +
                              "message in Flask log and browser developer tools.");
                    }
                });
                """)
    #endregion

    #region Plot3
    in_stock = av_data[av_data.availability == 'In Stock'].groupby('occasion').availability.count().reset_index()
    total = av_data.groupby('occasion').availability.count().reset_index()
    total.rename(columns={'availability': 'total'}, inplace=True)
    in_stock = in_stock.merge(total, how='left', on='occasion')
    in_stock['Stock %'] = in_stock['availability'] / in_stock['total']
    av_plot3_df = in_stock[['occasion', 'Stock %']]
    av_plot3_x = av_plot3_df['occasion']
    av_plot3_y = av_plot3_df['Stock %']
    av_plot3_source = ColumnDataSource(data=dict(x=av_plot3_x, y=av_plot3_y))
    av_plot3 = figure(x_range=av_plot3_x, plot_height=400, tools='save')
    av_plot3 = style_plot(av_plot3, 'Availability by Occasion')
    av_plot3.vbar(x='x',
               top='y',
               width=0.9,
               source=av_plot3_source,
               color="#c9d9d3")
    av_plot3.xgrid.grid_line_color = None
    av_plot3.y_range.start = 0
    av_plot3.xaxis.axis_label = "Occasion"
    av_plot3.yaxis.axis_label = "Stock %"
    av_plot3.xaxis.major_label_orientation = pi / 4
    av_plot3.yaxis.formatter = NumeralTickFormatter(format='0%')
    av_plot3.y_range.start = 0
    av_plot3.y_range.end = 1
    av_plot3.add_tools(HoverTool(
        tooltips=[
            ('Occasion', '@{x}'),
            ('Stock %', '@y{0.00%}'),
        ],
        mode='vline'
    ))
    av_plot3_callback = CustomJS(args=dict(source=av_plot3_source, controls=controls), code="""
                    var selected_value = new Object();

                    for (let key of Object.keys(controls)) {
                        var val = controls[key].value;
                        // console.log(key, val);
                        selected_value[key] = val;
                    }
                    //alert(Object.keys(controls));
                    var plot_data = source.data;

                    jQuery.ajax({
                        type: 'POST',
                        url: '/refresh_av3',
                        data: selected_value,
                        dataType: 'json',
                        success: function (json_from_server) {
                            plot_data.y = json_from_server.y;
                            plot_data.x = json_from_server.x; 
                            // source.data = json_from_server.data;
                            source.change.emit();
                        },
                        error: function() {
                            alert("Oh no, something went wrong. Search for an error " +
                                  "message in Flask log and browser developer tools.");
                        }
                    });
                    """)
    #endregion

    # region Callbacks
    for single_control in controls_array:
        single_control.js_on_change('value', av_plot1_callback)
        single_control.js_on_change('value', dt1_callback)
        single_control.js_on_change('value', av_plot2_callback)
        single_control.js_on_change('value', av_plot3_callback)
    #endregion

    #region Layout
    filters_row = row(*controls_array, background="#24282e", margin=(10,0,10,0))
    _layout = layout(children=[
        [filters_row],
        [av_plot1],
        [data_table1],
        [av_plot2, av_plot3]
    ],
        sizing_mode='stretch_width')
    script, div = components(_layout)
    #endregion

    return render_template('dashboard/availability.html',
                           script=script,
                           div=div,
                           js_resources=INLINE.render_js(),
                           css_resources=INLINE.render_css()
                           ).encode(encoding='UTF-8')

@dashboard.route('/dashboard/promotions')
@login_required
def promotions():

    # region Filters
    filters = create_filters(pr_data)
    controls = create_controls(filters)
    controls_array = get_controls_list(controls)
    # endregion

    # region Plot1
    pr_plot1_df = pr_data[pr_data.promotion_price > 0].groupby('month').promotion_price.count().reset_index()
    pr_plot1_df.month = pr_plot1_df.apply(lambda x: datetime.strptime(x.month, "%b").month, axis=1)
    pr_plot1_df = pr_plot1_df.sort_values(by='month')
    pr_plot1_df.month = pr_plot1_df.month.apply(lambda x: cal.month_abbr[x])
    pr_plot1_x = pr_plot1_df.month
    pr_plot1_y = pr_plot1_df.promotion_price
    pr_plot1_source = ColumnDataSource(data=dict(x=pr_plot1_x, y=pr_plot1_y))
    pr_plot1 = figure(x_range=pr_plot1_x, plot_height=400, tools='save')
    pr_plot1 = style_plot(pr_plot1, 'Count of Promotions by Month')
    pr_plot1.line('x', 'y', source=pr_plot1_source, line_width=3, line_alpha=0.6)
    pr_plot1.xaxis.axis_label = "Month"
    pr_plot1.yaxis.axis_label = "# Promotions"
    pr_plot1.xaxis.major_label_orientation = pi / 4
    pr_plot1.yaxis.formatter = NumeralTickFormatter(format='+0,0')
    pr_plot1.y_range.start=0
    pr_plot1.add_tools(HoverTool(
        tooltips=[
            ('Month', '@{x}'),
            ('Active Promotions', '@y{+0,0}'),
        ],
        mode='vline'
    ))
    pr_plot1_callback = CustomJS(args=dict(source=pr_plot1_source, controls=controls), code="""
            var selected_value = new Object();

            for (let key of Object.keys(controls)) {
                var val = controls[key].value;
                // console.log(key, val);
                selected_value[key] = val;
            }
            //alert(Object.keys(controls));
            var plot_data = source.data;

            jQuery.ajax({
                type: 'POST',
                url: '/refresh_pr1',
                data: selected_value,
                dataType: 'json',
                success: function (json_from_server) {
                    plot_data.y = json_from_server.y;
                    plot_data.x = json_from_server.x; 
                    // source.data = json_from_server.data;
                    source.change.emit();
                },
                error: function() {
                    alert("Oh no, something went wrong. Search for an error " +
                          "message in Flask log and browser developer tools.");
                }
            });
            """)
    # endregion

    #region DataTable1

    promo_price = pr_data.groupby('category').promotion_price.mean().reset_index()
    promo_price.rename(columns={'promotion_price':'avg_promotion_price'}, inplace=True)
    promo_count = pr_data[pr_data.promotion_price > 0].groupby('category').promotion_price.count().reset_index()
    promo_count.rename(columns={'promotion_price':'total_promotions'}, inplace=True)
    reg_price = pr_data.groupby('category').regular_price.mean().reset_index()
    reg_price.rename(columns={'regular_price':'avg_regular_price'}, inplace=True)
    data = promo_price.merge(promo_count, how='left', on='category')
    data = data.merge(reg_price, how='left', on='category')
    data['avg_discount'] = (data['avg_regular_price'] - data['avg_promotion_price'])/data['avg_regular_price']
    data['avg_promotion_price']  = data.apply(lambda x: human_readable(x['avg_promotion_price']), axis=1)
    data['avg_regular_price']  = data.apply(lambda x: human_readable(x['avg_regular_price']), axis=1)

    data = data.sort_values(by='avg_discount', ascending=False)
    dt_source = ColumnDataSource(data)
    columns = [
        TableColumn(field='category', title="Category"),
        TableColumn(field='total_promotions', title="Promotion Count", formatter=NumberFormatter(format='+0,0')),
        TableColumn(field='avg_regular_price', title="Avg Regular Price", formatter=HTMLTemplateFormatter(template=sales_val_template)),
        TableColumn(field='avg_promotion_price', title="Avg Promotion Price", formatter=HTMLTemplateFormatter(template=sales_val_template)),
        TableColumn(field='avg_discount', title="Avg Discount", formatter=NumberFormatter(format='0.00%')),
    ]
    data_table = DataTable(source=dt_source, columns=columns, height=300, index_position=None,
                           auto_edit=True, sizing_mode='stretch_width', background=None, css_classes=['bokeh-table']
                           )

    dt_callback = CustomJS(args=dict(source=dt_source, controls=controls), code="""
                var selected_value = new Object();

                for (let key of Object.keys(controls)) {
                    var val = controls[key].value;
                    // console.log(key, val);
                    selected_value[key] = val;
                }
                //alert(Object.keys(controls));
                var plot_data = source.data;

                jQuery.ajax({
                    type: 'POST',
                    url: '/refresh_pr2',
                    data: selected_value,
                    dataType: 'json',
                    success: function (json_from_server) {
                        plot_data.category = json_from_server.category;
                        plot_data.total_promotions = json_from_server.total_promotions; 
                        plot_data.avg_regular_price = json_from_server.avg_regular_price; 
                        plot_data.avg_promotion_price = json_from_server.avg_promotion_price; 
                        plot_data.avg_discount = json_from_server.avg_discount; 
                        source.change.emit();
                    },
                    error: function() {
                        alert("Oh no, something went wrong. Search for an error " +
                              "message in Flask log and browser developer tools.");
                    }
                });
                """)
    #endregion

    #region Callbacks
    for single_control in controls_array:
        single_control.js_on_change('value', pr_plot1_callback)
        single_control.js_on_change('value', dt_callback)

    #endregion

    #region Layout
    filters_row = row(*controls_array, background="#24282e", margin=(10,0,10,0))
    _layout = layout(children=[
        [filters_row],
        [pr_plot1],
        [data_table],
    ],
        sizing_mode='stretch_width')
    script, div = components(_layout)

    #endregion

    return render_template('dashboard/promotions.html',
                           script=script,
                           div=div,
                           js_resources=INLINE.render_js(),
                           css_resources=INLINE.render_css()
                           ).encode(encoding='UTF-8')

@dashboard.route('/dashboard/reviews')
@login_required
def reviews():
    # region Filters
    for col in ['category', 'occasion', 'brand', 'collection', 'prod_material', 'gender', 'age_group']:
        review_data[col] = review_data[col].fillna('Unknown')
    filters = create_filters(review_data)  # Create the filters
    controls = create_controls(filters)  # Create the Controls for filters
    controls_array = get_controls_list(controls)
    # endregion

    # region Plot1
    t_rev = review_data.shape[0]
    re_plot1_df = review_data.groupby('gender').rating.count().reset_index()
    re_plot1_df.rating = re_plot1_df.rating/t_rev

    re_plot1_x = re_plot1_df['gender']
    re_plot1_y = re_plot1_df['rating']
    re_plot1_source = ColumnDataSource(data=dict(x=re_plot1_x, y=re_plot1_y))
    re_plot1 = figure(x_range=re_plot1_x, plot_height=400, tools='save')
    re_plot1 = style_plot(re_plot1, 'Gender Distribution')
    re_plot1.vbar(x='x',
                  top='y',
                  width=0.9,
                  source=re_plot1_source,
                  color="#c9d9d3")
    re_plot1.xgrid.grid_line_color = None
    re_plot1.y_range.start = 0
    re_plot1.xaxis.axis_label = "Gender"
    re_plot1.yaxis.axis_label = "# Reviews"
    re_plot1.xaxis.major_label_orientation = pi / 4
    re_plot1.yaxis.formatter = NumeralTickFormatter(format='0%')
    re_plot1.y_range.start = 0
    re_plot1.y_range.end = 1
    re_plot1.add_tools(HoverTool(
        tooltips=[
            ('Gender', '@{x}'),
            ('Review %', '@y{0.00%}'),
        ],
        mode='vline'
    ))
    re_plot1_callback = CustomJS(args=dict(source=re_plot1_source, controls=controls), code="""
                        var selected_value = new Object();

                        for (let key of Object.keys(controls)) {
                            var val = controls[key].value;
                            // console.log(key, val);
                            selected_value[key] = val;
                        }
                        //alert(Object.keys(controls));
                        var plot_data = source.data;

                        jQuery.ajax({
                            type: 'POST',
                            url: '/refresh_re1',
                            data: selected_value,
                            dataType: 'json',
                            success: function (json_from_server) {
                                plot_data.y = json_from_server.y;
                                plot_data.x = json_from_server.x; 
                                // source.data = json_from_server.data;
                                source.change.emit();
                            },
                            error: function() {
                                alert("Oh no, something went wrong. Search for an error " +
                                      "message in Flask log and browser developer tools.");
                            }
                        });
                        """)
    # endregion

    # region Plot2
    re_plot2_df = review_data.groupby('review_source').rating.count().reset_index()
    re_plot2_df.rating = re_plot2_df.rating / t_rev

    re_plot2_x = re_plot2_df['review_source']
    re_plot2_y = re_plot2_df['rating']
    re_plot2_source = ColumnDataSource(data=dict(x=re_plot2_x, y=re_plot2_y))
    re_plot2 = figure(x_range=re_plot2_x, plot_height=400, tools='save')
    re_plot2 = style_plot(re_plot2, 'Source Distribution')
    re_plot2.vbar(x='x',
                  top='y',
                  width=0.9,
                  source=re_plot2_source,
                  color="#c9d9d3")
    re_plot2.xgrid.grid_line_color = None
    re_plot2.y_range.start = 0
    re_plot2.xaxis.axis_label = "Source"
    re_plot2.yaxis.axis_label = "# Reviews"
    re_plot2.xaxis.major_label_orientation = pi / 4
    re_plot2.yaxis.formatter = NumeralTickFormatter(format='0%')
    re_plot2.y_range.start = 0
    re_plot2.y_range.end = 1
    re_plot2.add_tools(HoverTool(
        tooltips=[
            ('Source', '@{x}'),
            ('Review %', '@y{0.00%}'),
        ],
        mode='vline'
    ))
    re_plot2_callback = CustomJS(args=dict(source=re_plot2_source, controls=controls), code="""
                            var selected_value = new Object();

                            for (let key of Object.keys(controls)) {
                                var val = controls[key].value;
                                // console.log(key, val);
                                selected_value[key] = val;
                            }
                            //alert(Object.keys(controls));
                            var plot_data = source.data;

                            jQuery.ajax({
                                type: 'POST',
                                url: '/refresh_re2',
                                data: selected_value,
                                dataType: 'json',
                                success: function (json_from_server) {
                                    plot_data.y = json_from_server.y;
                                    plot_data.x = json_from_server.x; 
                                    // source.data = json_from_server.data;
                                    source.change.emit();
                                },
                                error: function() {
                                    alert("Oh no, something went wrong. Search for an error " +
                                          "message in Flask log and browser developer tools.");
                                }
                            });
                            """)
    # endregion

    # region Plot3
    review_data.loc[review_data.rating <= 3, 'review_score'] = 'negative'
    review_data.loc[review_data.rating > 3, 'review_score'] = 'positive'
    re_plot3_df = review_data.groupby('review_score').rating.count().reset_index()
    re_plot3_df.rating = re_plot3_df.rating / t_rev

    re_plot3_x = re_plot3_df['review_score']
    re_plot3_y = re_plot3_df['rating']
    re_plot3_source = ColumnDataSource(data=dict(x=re_plot3_x, y=re_plot3_y))
    re_plot3 = figure(x_range=re_plot3_x, plot_height=400, tools='save')
    re_plot3 = style_plot(re_plot3, 'Score Distribution')
    re_plot3.vbar(x='x',
                  top='y',
                  width=0.9,
                  source=re_plot3_source,
                  color="#c9d9d3")
    re_plot3.xgrid.grid_line_color = None
    re_plot3.y_range.start = 0
    re_plot3.xaxis.axis_label = "Score"
    re_plot3.yaxis.axis_label = "# Reviews"
    re_plot3.xaxis.major_label_orientation = pi / 4
    re_plot3.yaxis.formatter = NumeralTickFormatter(format='0%')
    re_plot3.y_range.start = 0
    re_plot3.y_range.end = 1
    re_plot3.add_tools(HoverTool(
        tooltips=[
            ('Score', '@{x}'),
            ('Review %', '@y{0.00%}'),
        ],
        mode='vline'
    ))
    re_plot3_callback = CustomJS(args=dict(source=re_plot3_source, controls=controls), code="""
                                var selected_value = new Object();

                                for (let key of Object.keys(controls)) {
                                    var val = controls[key].value;
                                    // console.log(key, val);
                                    selected_value[key] = val;
                                }
                                //alert(Object.keys(controls));
                                var plot_data = source.data;

                                jQuery.ajax({
                                    type: 'POST',
                                    url: '/refresh_re3',
                                    data: selected_value,
                                    dataType: 'json',
                                    success: function (json_from_server) {
                                        plot_data.y = json_from_server.y;
                                        plot_data.x = json_from_server.x; 
                                        // source.data = json_from_server.data;
                                        source.change.emit();
                                    },
                                    error: function() {
                                        alert("Oh no, something went wrong. Search for an error " +
                                              "message in Flask log and browser developer tools.");
                                    }
                                });
                                """)
    # endregion

    # region Plot4
    #review_data['rating_hr'] = review_data.apply(lambda x: str(x['rating']) + ' stars', axis=1)
    review_data['rating_hr'] = [str(ele) + ' starts' for ele in review_data['rating']]
    re_plot4_df = review_data.groupby('rating_hr').review_source.count().reset_index()
    re_plot4_df.review_source = re_plot4_df.review_source / t_rev

    re_plot4_x = re_plot4_df['rating_hr']
    re_plot4_y = re_plot4_df['review_source']
    re_plot4_source = ColumnDataSource(data=dict(x=re_plot4_x, y=re_plot4_y))
    re_plot4 = figure(x_range=re_plot4_x, plot_height=400, tools='save')
    re_plot4 = style_plot(re_plot4, 'Star Rating Distribution')
    re_plot4.vbar(x='x',
                  top='y',
                  width=0.9,
                  source=re_plot4_source,
                  color="#c9d9d3")
    re_plot4.xgrid.grid_line_color = None
    re_plot4.y_range.start = 0
    re_plot4.xaxis.axis_label = " Star Rating"
    re_plot4.yaxis.axis_label = "# Reviews"
    re_plot4.xaxis.major_label_orientation = pi / 4
    re_plot4.yaxis.formatter = NumeralTickFormatter(format='0%')
    re_plot4.y_range.start = 0
    re_plot4.y_range.end = 1
    re_plot4.add_tools(HoverTool(
        tooltips=[
            ('Rating', '@{x}'),
            ('Review %', '@y{0.00%}'),
        ],
        mode='vline'
    ))
    re_plot4_callback = CustomJS(args=dict(source=re_plot4_source, controls=controls), code="""
                                    var selected_value = new Object();

                                    for (let key of Object.keys(controls)) {
                                        var val = controls[key].value;
                                        // console.log(key, val);
                                        selected_value[key] = val;
                                    }
                                    //alert(Object.keys(controls));
                                    var plot_data = source.data;

                                    jQuery.ajax({
                                        type: 'POST',
                                        url: '/refresh_re4',
                                        data: selected_value,
                                        dataType: 'json',
                                        success: function (json_from_server) {
                                            plot_data.y = json_from_server.y;
                                            plot_data.x = json_from_server.x; 
                                            // source.data = json_from_server.data;
                                            source.change.emit();
                                        },
                                        error: function() {
                                            alert("Oh no, something went wrong. Search for an error " +
                                                  "message in Flask log and browser developer tools.");
                                        }
                                    });
                                    """)
    # endregion

    # region Plot5
    re_plot5_df = review_data.groupby('age_group').rating.count().reset_index()
    re_plot5_df.rating = re_plot5_df.rating / t_rev

    re_plot5_x = re_plot5_df['age_group']
    re_plot5_y = re_plot5_df['rating']
    re_plot5_source = ColumnDataSource(data=dict(x=re_plot5_x, y=re_plot5_y))
    re_plot5 = figure(x_range=[str(ele) for ele in re_plot5_x], plot_height=400, tools='save')
    re_plot5 = style_plot(re_plot5, 'Reviews by Age')
    re_plot5.vbar(x='x',
                  top='y',
                  width=0.9,
                  source=re_plot5_source,
                  color="#c9d9d3")
    re_plot5.xgrid.grid_line_color = None
    re_plot5.y_range.start = 0
    re_plot5.xaxis.axis_label = "Age Group"
    re_plot5.yaxis.axis_label = "# Reviews"
    re_plot5.xaxis.major_label_orientation = pi / 4
    re_plot5.yaxis.formatter = NumeralTickFormatter(format='0%')
    re_plot5.y_range.start = 0
    re_plot5.y_range.end = 1
    re_plot5.add_tools(HoverTool(
        tooltips=[
            ('Age', '@{x}'),
            ('Review %', '@y{0.00%}'),
        ],
        mode='vline'
    ))
    re_plot5_callback = CustomJS(args=dict(source=re_plot5_source, controls=controls), code="""
                                        var selected_value = new Object();

                                        for (let key of Object.keys(controls)) {
                                            var val = controls[key].value;
                                            // console.log(key, val);
                                            selected_value[key] = val;
                                        }
                                        //alert(Object.keys(controls));
                                        var plot_data = source.data;

                                        jQuery.ajax({
                                            type: 'POST',
                                            url: '/refresh_re5',
                                            data: selected_value,
                                            dataType: 'json',
                                            success: function (json_from_server) {
                                                plot_data.y = json_from_server.y;
                                                plot_data.x = json_from_server.x; 
                                                // source.data = json_from_server.data;
                                                source.change.emit();
                                            },
                                            error: function() {
                                                alert("Oh no, something went wrong. Search for an error " +
                                                      "message in Flask log and browser developer tools.");
                                            }
                                        });
                                        """)
    # endregion

    # region DataTable
    rev_df = review_data[['product_id', 'review_date', 'image_url', 'review_title', 'review_text', 'product_title', 'rating']].copy()
    rev_df.review_date = pd.to_datetime(rev_df.review_date).dt.strftime('%d-%m-%Y')
    rev_df.fillna('', inplace=True)
    rev_df.loc[rev_df.image_url == '', 'image_url'] = 'https://i1.wp.com/fremontgurdwara.org/wp-content/uploads/2020/06/no-image-icon-2.png'
    rev_df.loc[rev_df.product_title == '', 'product_title'] = 'Product Unknown'
    rev_df.loc[rev_df.product_id == '', 'product_id'] = 0
    rev_df = rev_df.sort_values(by='review_date', ascending=False)

    dt_source = ColumnDataSource(rev_df)

    columns = [
        TableColumn(field='review_date', title="Date", formatter=HTMLTemplateFormatter(template=text_template)),
        TableColumn(field='product_title', title="Product", formatter=HTMLTemplateFormatter(template=pr_title_template)),
        TableColumn(field='review_title', title="Review Summary", formatter=HTMLTemplateFormatter(template=text_template)),
        TableColumn(field='review_text', title="Review Text", formatter=HTMLTemplateFormatter(template=text_template)),
        TableColumn(field='rating', title="Rating", formatter=HTMLTemplateFormatter(template=text_template)),
    ]
    data_table = DataTable(source=dt_source, columns=columns, height=500, row_height=50, index_position=None,
                           sizing_mode='stretch_width', background=None, css_classes=['bokeh-table'])
    re_dt_callback = CustomJS(args=dict(source=dt_source, controls=controls), code="""
               var selected_value = new Object();

               for (let key of Object.keys(controls)) {
                   var val = controls[key].value;
                   // console.log(key, val);
                   selected_value[key] = val;
               }
               //alert(Object.keys(controls));
               var plot_data = source.data;

               jQuery.ajax({
                   type: 'POST',
                   url: '/refresh_re6',
                   data: selected_value,
                   dataType: 'json',
                   success: function (json_from_server) {
                       plot_data.review_date = json_from_server.review_date;
                       plot_data.product_title = json_from_server.product_title; 
                       plot_data.product_id = json_from_server.product_id; 
                       plot_data.review_title = json_from_server.review_title; 
                       plot_data.review_text = json_from_server.review_text; 
                       plot_data.rating = json_from_server.rating; 
                       plot_data.image_url = json_from_server.image_url; 
                       source.change.emit();
                   },
                   error: function() {
                       alert("Oh no, something went wrong. Search for an error " +
                             "message in Flask log and browser developer tools.");
                   }
               });
               """)
    # endregion

    #region WordCloud
    stopwords = set(STOPWORDS)
    text = ' '.join(r for r in review_data.review_text)
    wordcloud = WordCloud(stopwords=stopwords).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/img/cloud.png')

    wcdata=review_data.loc[:0, :]
    wcdata['cloud']='http://127.0.0.1:5000/static/img/cloud.png'
    wc_source=ColumnDataSource(wcdata)

    columns = [
        TableColumn(field='cloud', title="Customer Sentiment",
                    formatter=HTMLTemplateFormatter(template="""<img src="<%= cloud%>" style="width:400px;height:290px;border:0">""")),
    ]
    wc_table = DataTable(source=wc_source, columns=columns, height=300, width=410, row_height=250, index_position=None,
                          sizing_mode='fixed', background=None, css_classes=['bokeh-table'])

    #endregion

    for single_control in controls_array:
        single_control.js_on_change('value', re_plot1_callback)
        single_control.js_on_change('value', re_plot2_callback)
        single_control.js_on_change('value', re_plot3_callback)
        single_control.js_on_change('value', re_plot4_callback)
        single_control.js_on_change('value', re_plot5_callback)
        single_control.js_on_change('value', re_dt_callback)

    filters_row = row(*controls_array, background="#24282e", margin=(10, 0, 10, 0))
    _layout = layout(children=[
        [filters_row],
        [re_plot1, re_plot2, re_plot3],
        [re_plot4, re_plot5],
        [data_table],
        [wc_table]
    ],
        sizing_mode='stretch_width')
    script, div = components(_layout)

    return render_template('dashboard/reviews.html',
                           script=script,
                           div=div,
                           js_resources=INLINE.render_js(),
                           css_resources=INLINE.render_css()
                           ).encode(encoding='UTF-8')

@dashboard.route('/dashboard/products', methods=['GET', 'POST'])
@login_required
def products():
    if request.method=='POST':
        data = ImmutableMultiDict(request.values).to_dict(flat=False)
        query =f"""UPDATE product SET
        title='{data['product_title'][0]}',
        description='{data['description'][0]}',
        recipient='{data['recipient'][0]}',
        image_url='{data['image_url'][0]}'
        WHERE id='{data['product_id'][0]}'
        """
        alter_db(query)
        #print(data['product_id'][0])

    pdata = query_db("""SELECT id as product_id, image_url, title as product_title, description, recipient
                    FROM product;""")

    source=ColumnDataSource(pdata)
    columns=[
        TableColumn(field="product_title", title="Product Title", editor=TextEditor(), formatter=HTMLTemplateFormatter(template=pr_title_template)),
        TableColumn(field="description", title="Description", editor=TextEditor(), formatter=HTMLTemplateFormatter(template=text_template)),
        TableColumn(field="recipient", title="Recipient", editor=StringEditor(), formatter=HTMLTemplateFormatter(template=text_template)),
        TableColumn(field="image_url", title="Image", editor=StringEditor(), formatter=HTMLTemplateFormatter(template=text_template)),
    ]
    datatable = DataTable(height=800, source=source, columns=columns, index_position=None, sizing_mode='stretch_width',
                          row_height=50, editable=True, selectable=True, css_classes=['bokeh-table'])

    source.js_on_change('patching', CustomJS(args=dict(source=source), code="""
               var changed_product = new Object();
               var selected_index = source.selected.indices[0];               
               changed_product['product_id'] = source.data.product_id[selected_index]; 
               changed_product['product_title'] = source.data.product_title[selected_index]; 
               changed_product['description'] = source.data.description[selected_index]; 
               changed_product['recipient'] = source.data.recipient[selected_index]; 
               changed_product['image_url'] = source.data.image_url[selected_index]; 
               
               jQuery.ajax({
                  type: 'POST',
                  url: '/dashboard/products',
                  data: changed_product,
                  dataType: 'json'                  
               });
               """))

    _layout = layout(children=[
        [datatable],
    ],sizing_mode='stretch_width')
    script, div = components(_layout)

    return render_template('dashboard/products.html',
                           script=script,
                           div=div,
                           js_resources=INLINE.render_js(),
                           css_resources=INLINE.render_css()
                           ).encode(encoding='UTF-8')

@dashboard.route('/dashboard/settings', methods=['GET', 'POST'])
@login_required
def settings():
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

    return render_template('dashboard/settings.html', form=form)

@dashboard.route('/dashboard/calendar')
@login_required
def calendar():
    return render_template('dashboard/calendar.html')

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

        data = {'product': prod_data, 'availability': av_data, 'sales': s_data,
                'promotions': p_data, 'reviews': r_data}

        return render_template('dashboard/details.html', data=data, id=id)
    else:
        return redirect(url_for('dashboard.not_found'))

@dashboard.route('/404')
def not_found():
    return render_template('404.html')

#endregion



