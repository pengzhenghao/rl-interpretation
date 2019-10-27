google.charts.load('current', {'packages': ['corechart', 'controls']});
google.charts.setOnLoadCallback(drawChart);

var data_table;
// var current_data_view;
var current_tune_flag = true;
var current_tensity = 0;
// var current_method = null;

var rawData = $.parseJSON($.ajax({
    url: 'test_data.json',
    dataType: "json",
    async: false
}).responseText);

function createCustomHTMLContent(videoPath) {
    return "" +
        // '<a href="' + hyperLink + '">' +
        '<video width="80" height="80" autoplay loop muted>' +
        '<source src="' + videoPath +
        '" type="video/mp4" /></video>'
    // '</a>' +
}

function changeText(elementId, text) {
    var element = document.getElementById(elementId);
    element.innerText = text;
}

// var filter;
// var count = 0;

// var chart;

function drawChart() {
    ////////// Step 1: Initialize The figure //////////
    const figure_info = rawData['figure_info'];
    const tensity_multiplier = figure_info['tensity_multiplier'];

    var dashboard, slider, chart, filter;

    function build_dashboard() {
        chart = new google.visualization.ChartWrapper(
            {
                "chartType": "ScatterChart",
                "containerId": "chart_div",
                "options": {
                    "tooltip": {"isHtml": true, "trigger": "selection"},
                    "title": figure_info['title'],
                    "hAxis": {
                        "title": figure_info['xlabel'],
                        "minValue": figure_info['xlim'] ? figure_info['xlim'][0] : null,
                        "maxValue": figure_info['xlim'] ? figure_info['xlim'][1] : null
                    },
                    "vAxis": {
                        "title": figure_info['ylabel'],
                        "minValue": figure_info['ylim'] ? figure_info['ylim'][0] : null,
                        "maxValue": figure_info['ylim'] ? figure_info['ylim'][1] : null
                    },
                    "legend": "none",
                    "aggregationTarget": "none",
                    "selectionMode": "multiple",
                    "theme": "maximized",
                    "fontSize": 12
                },
                "view": {'columns': [0, 1]}
                // "view": {
                //     'columns': [0, 1],
                //     'rows': data_table.getFilteredRows(
                //         [{
                //             column: data_table.getColumnIndex("tensity"),
                //             value: current_tensity * tensity_multiplier
                //         }],
                //         [{
                //             column: data_table.getColumnIndex("method"),
                //             value: rawData['figure_info']['methods'][0]
                //         }]
                //     )
                // }
            });
        filter = new google.visualization.ControlWrapper({
            'controlType': 'CategoryFilter',
            'containerId': 'control_div',
            'options': {
                'filterColumnLabel': "method",
                // 'filterColumnIndex': 4,
                'ui': {
                    // 'label': 'Choose one of the Representation Methods: ',
                    'allowNone': false,
                    "allowMultiple": false,
                    "allowTyping": false,
                    // "labelStacking": 'vertical',
                    // 'cssClass': 'step1'
                }
            },
            'state': {'selectedValues': [rawData['figure_info']['methods'][0]]}
        });
        dashboard = new google.visualization.Dashboard(
            document.getElementById('dashboard_div'));
        dashboard.bind(filter, chart);
        dashboard.draw(data_table);
        // google.visualization.events.addOneTimeListener(chart, 'ready', function () {
        //     console.log("READY!!!!!!!!!!!");
        //     flush();
        // });
    }

    function get_exact_std(slider_value) {
        return figure_info['tensities'][slider_value] / tensity_multiplier
    }

// Create the slider for std changing
    function build_slider() {
        slider = document.getElementById("tensitySlider");
        slider.value = figure_info['tensities'][0];
        slider.min = 0;
        slider.max = figure_info['num_tensities'] - 1;
        slider.oninput = function () {
            current_tensity = get_exact_std(this.value);
            flush();
        };
        var dis_str = get_exact_std(figure_info['tensities'][0]).toString();
        for (var t = 1; t < figure_info['tensities'].length; t++) {
            dis_str = dis_str + ", " + get_exact_std(t).toString();
        }
        current_tensity = "All of: \n" + dis_str;
        changeText("title_of_table", rawData['web_info']['title']);
        changeText("introduction", rawData['web_info']['introduction']);
        changeText("tensity", current_tensity);
        changeText("tensity2", "all");
        changeText("update_date", rawData['web_info']['update_date']);
    }

////////// Step 2: Define some useful function //////////
    function setup_data_table() {
        var newData;
        if (current_tune_flag) {
            newData = rawData['data']['fine_tuned']
        } else {
            newData = rawData['data']['not_fine_tuned']
        }
        data_table = new google.visualization.DataTable(newData);

        // Fill the tooltip
        data_table.addColumn(
            {
                'type': 'string',
                'role': 'tooltip',
                'p': {'html': true}
            }
        );

        var url, cell_html, row, extra;
        for (row = 0; row < data_table.getNumberOfRows(); row++) {
            url = data_table.getRowProperty(row, "url");
            if (url === null) {
                cell_html = '<p>No Video Provided</p>';
            } else {
                cell_html = createCustomHTMLContent(url)
            }
            extra = data_table.getRowProperty(row, "extra");
            if (extra !== null) {
                cell_html = cell_html +
                    '<br><p style="max-width: 80px;' +
                    'word-wrap: break-word">' + extra + '</p>';
            }
            cell_html = '<dev style="padding:0 0 0 0">' + cell_html
                + '</dev>';
            data_table.setCell(row, data_table.getNumberOfColumns() - 1, cell_html);
        }
    }

    function flush() {
        update_data_view();
        changeText("tensity", current_tensity);
        changeText("tensity2", current_tensity);
        changeText("finetuned", current_tune_flag ? "fine-tuned" : "not fine-tuned");
        dashboard.draw(data_table);
    }

    function update_data_view() {
        var tmp_dt = chart.getDataTable();
        if (tmp_dt !== null) {
            chart.setView({
                "columns": [0, 1],
                "rows": tmp_dt.getFilteredRows(
                    [{
                        column: tmp_dt.getColumnIndex("tensity"),
                        value: current_tensity * tensity_multiplier
                    }]
                )
            });
        }
    }

    function init() {
        build_slider();
        setup_data_table();
        build_dashboard();
    }

    init();

////////// Event Handler //////////

// Change method
//     google.visualization.events.addListener(filter, 'statechange', function () {
//         console.log("CHANEG METHOD!!!!!!!!!!!");
//
//         chart.setOption("vAxis.minValue", 0);
//         chart.setOption("vAxis.maxValue", 0);
//         chart.setOption("hAxis.minValue", 0);
//         chart.setOption("hAxis.maxValue", 0);
//
//         flush();
//     });

    change2FineTuned = function () {
        current_tune_flag = true;
        init();
    };

    change2NotFineTuned = function () {
        current_tune_flag = false;
        init;
    };

    reset_slider = function () {
        init();
    };

    clear_selection = function () {
        chart.getChart().setSelection();
    }

}