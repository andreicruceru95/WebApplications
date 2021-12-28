from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.datastructures import ImmutableMultiDict
from flask_login import login_required, current_user
from datetime import datetime
from PIL import Image
from bokeh.models import ColumnDataSource, MultiSelect, FuncTickFormatter, HoverTool, DataTable, TableColumn, NumberFormatter
from bokeh.models.callbacks import CustomJS
from bokeh.resources import INLINE, CDN
from bokeh.embed import components, file_html, json_item
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, layout
from bokeh.palettes import Category20c
from bokeh.transform import cumsum, dodge
from numerize import numerize
from math import pi

import mysql.connector
import pandas as pd
import secrets
import calendar as cal
import store
import json
import os


dashboard = Blueprint('dashboard', __name__)

from app import db, app
from models import UpdateUserInfo


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

_colors =['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476', '#a1d99b']
sales_data = query_db("""SELECT p.category, p.occasion, p.prod_material, p.brand, p.collection, s.sales, d.month, d.quarter
                    FROM sales s
                    LEFT JOIN product p ON s.product_id = p.id
                    LEFT JOIN date d on s.date_id = d.id LIMIT 10000""");


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
    print(df.month)
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

########################################################################################################################
# SALES PAGE

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

def filter_sales_data(data):
    df = sales_data[(sales_data['category'].isin(data['category[]']) &
                     sales_data['occasion'].isin(data['occasion[]']) &
                     sales_data['collection'].isin(data['collection[]']) &
                     sales_data['prod_material'].isin(data['material[]']) &
                     sales_data['brand'].isin(data['brand[]']) &
                     sales_data['month'].isin(data['month[]']))]

    return df

# region Sales Refresh
@dashboard.route("/refresh_sales0", methods=['POST'])
def refresh_sales_0():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_sales_data(data)
    t_sales = df.groupby('brand').sales.sum().reset_index()
    t_sales.rename(columns={'sales': 'total_sales'}, inplace=True)
    q_sales = df.groupby(['brand', 'quarter']) \
        .sales.sum() \
        .reset_index() \
        .pivot(
        values='sales',
        index='brand',
        columns='quarter'
    )
    data = q_sales.merge(t_sales, how='left', on='brand')
    data = data.sort_values(by='total_sales', ascending=False)

    brand = list(data.brand)
    q1 = list(data['2021-Q1'])
    q2 = list(data['2021-Q2'])
    q3 = list(data['2021-Q3'])
    q4 = list(data['2021-Q4'])
    t = list(data['total_sales'])
    response = jsonify({'brand':brand, 'q1':q1, 'q2':q2, 'q3':q3, 'q4':q4, 't':t})
    return response

@dashboard.route("/refresh_sales1", methods=['POST'])
def refresh_sales_1():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_sales_data(data)

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
    df = filter_sales_data(data)

    df = df.groupby('category').sales.sum().reset_index()
    df = df.sort_values(by='sales')

    x = list(df.sales)
    y = list(df.category)
    response = jsonify({'x': x, 'y': y})
    return response

@dashboard.route("/refresh_sales3", methods=['POST'])
def refresh_sales_3():
    data = ImmutableMultiDict(request.values).to_dict(flat=False)
    df = filter_sales_data(data)
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
    df = filter_sales_data(data)
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
    df = filter_sales_data(data)

    df = df.groupby('occasion').sales.sum().reset_index()

    df = df.sort_values(by='sales')
    x = list(df.occasion)
    y = list(df.sales)
    response = jsonify({'x': x, 'y': y})
    return response

#endregion

@dashboard.route('/dashboard/sales')
@login_required
def sales():
    #region Filters
    filters = create_filters(sales_data) # Create the filters
    controls = create_controls(filters) # Create the Controls for filters
    controls_array = get_controls_list(controls)
    #endregion

    # region DataTable
    t_sales = sales_data.groupby('brand').sales.sum().reset_index()
    t_sales.rename(columns={'sales': 'total_sales'}, inplace=True)
    q_sales = sales_data.groupby(['brand', 'quarter']) \
        .sales.sum() \
        .reset_index() \
        .pivot(
        values='sales',
        index='brand',
        columns='quarter'
    )
    data = q_sales.merge(t_sales, how='left', on='brand')
    data = data.sort_values(by='total_sales', ascending=False)
    dt_source = ColumnDataSource(data)
    columns = [
        TableColumn(field='brand', title="Brand"),
        TableColumn(field='2021-Q1', title="Q1", formatter=NumberFormatter(format='$0,0.00')),
        TableColumn(field='2021-Q2', title="Q2", formatter=NumberFormatter(format='$0,0.00')),
        TableColumn(field='2021-Q3', title="Q3", formatter=NumberFormatter(format='$0,0.00')),
        TableColumn(field='2021-Q4', title="Q4", formatter=NumberFormatter(format='$0,0.00')),
        TableColumn(field='total_sales', title="Total Sales", formatter=NumberFormatter(format='$0,0.00')),
    ]
    data_table = DataTable(source=dt_source, columns=columns, height=500, index_position=None,
                           auto_edit=True, sizing_mode='stretch_width', background=None,
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
                url: '/refresh_sales0',
                data: selected_value,
                dataType: 'json',
                success: function (json_from_server) {
                    plot_data.brand = json_from_server.brand;
                    plot_data.total_sales = json_from_server.t; 
                    plot_data["2021-Q1"] = json_from_server.q1; 
                    plot_data["2021-Q2"] = json_from_server.q2; 
                    plot_data["2021-Q3"] = json_from_server.q3; 
                    plot_data["2021-Q4"] = json_from_server.q4; 
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
        line_color="white", fill_color='color', legend='category', source=plot3_source)
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
                line_color="white", fill_color='color', legend='quarter', source=plot4_source)
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

    for single_control in controls_array:
        single_control.js_on_change('value', plot1_callback)
        single_control.js_on_change('value', plot2_callback)
        single_control.js_on_change('value', plot3_callback)
        single_control.js_on_change('value', plot4_callback)
        single_control.js_on_change('value', plot5_callback)
        single_control.js_on_change('value', dt_callback)

    filters_row = row(*controls_array, background="#24282e", margin=(10,0,10,0))
    _layout = layout(children=[
            [filters_row],
            [plot2, plot5],
            [plot1],
            [data_table],
            [plot3, plot4],
        ],
    sizing_mode='stretch_width')
    script, div = components(_layout)

    return render_template('dashboard/sales.html',
                           script=script,
                           div=div,
                           js_resources = INLINE.render_js(),
                           css_resources=INLINE.render_css()
                           ).encode(encoding='UTF-8')


########################################################################################################################


@dashboard.route('/dashboard/availability')
@login_required
def availability():
    return render_template('dashboard/availability.html')


@dashboard.route('/dashboard/promotions')
@login_required
def promotions():
    return render_template('dashboard/promotions.html')


@dashboard.route('/dashboard/reviews')
@login_required
def reviews():
    return render_template('dashboard/reviews.html')

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
    data = {'products': product_data}
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
        return render_template('dashboard/details.html', data=data)
    else:
        return redirect(url_for('not_found'))


@dashboard.route('/404')
def not_found():
    return render_template('404.html')





