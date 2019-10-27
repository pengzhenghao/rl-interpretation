google.charts.load('current', {'packages': ['corechart', 'controls']});
google.charts.setOnLoadCallback(drawChart);

var data_table;
var current_data_view;
var current_tune_flag = true;
var current_tensity = 0;

var rawData = $.parseJSON($.ajax({
    url: 'test_data2.json',
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

// function reorganizeDataTable(data_table) {
//     data_table.
// }


function drawChart() {

    var figure_info = rawData['figure_info'];
    // Create the chart
    // var chart = new google.visualization.ScatterChart(document.getElementById('chart_div'));
    // var options = {
    //     tooltip: {isHtml: true, trigger: 'selection'},
    //     title: figure_info['title'],
    //     hAxis: {
    //         title: figure_info['xlabel'],
    //         minValue: figure_info['xlim'] ? figure_info['xlim'][0] : null,
    //         maxValue: figure_info['xlim'] ? figure_info['xlim'][1] : null
    //     },
    //     vAxis: {
    //         title: figure_info['ylabel'],
    //         minValue: figure_info['ylim'] ? figure_info['ylim'][0] : null,
    //         maxValue: figure_info['ylim'] ? figure_info['ylim'][1] : null
    //     },
    //     legend: 'none',
    //     aggregationTarget: 'none',
    //     selectionMode: 'multiple'
    // };

    var dashboard = new google.visualization.Dashboard(
        document.getElementById('dashboard_div'));

    var filter = new google.visualization.ControlWrapper({
        'controlType': 'CategoryFilter',
        'containerId': 'control_div',
        'options': {
            // 'filterColumnLabel': "x",
            'filterColumnIndex': 0,
            'ui': {
                'allowNone': true,
                "allowMultiple": true,
                "allowTyping": false
            }
        }
    });

    var chart = new google.visualization.ChartWrapper(
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
            }
        }
    );


    function get_exact_std(slider_value) {
        return slider_value / (figure_info['num_std'] - 1)
    }

    // Create the slider for std changing
    var slider = document.getElementById("tensitySlider");
    slider.value = figure_info['std_min'];
    slider.min = figure_info['std_min'];
    slider.max = figure_info['num_std'] - 1;
    slider.oninput = function () {
        current_tensity = get_exact_std(this.value);
        flush();
    };

    changeText("title_of_table", rawData['web_info']['title']);
    changeText("introduction", rawData['web_info']['introduction']);
    changeText("tensity", slider.value);
    changeText("tensity2", slider.value);
    changeText("update_date", rawData['web_info']['update_date']);

    setup_data_table();

    function setup_data_table() {
        var newData;
        var std = current_tensity;
        if (current_tune_flag) {
            newData = rawData['data']['fine_tuned']['fft']
        } else {
            newData = rawData['data']['not_fine_tuned']['fft']
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

        current_data_view = new google.visualization.DataView(data_table);
        current_data_view.setColumns([0, 1]);
    }

    function update_data_view() {
        // cloumn 3 is tensity
        current_data_view.setRows(
            data_table.getFilteredRows(
                [{column: 3, value: current_tensity}]
            )
        );
        return current_data_view;
    }

    function flush() {
        current_data_view = update_data_view();
        changeText("tensity", current_tensity);
        changeText("tensity2", current_tensity);
        changeText("finetuned", current_tune_flag ? "fine-tuned" : "not fine-tuned");
        dashboard.draw(current_data_view);
    }

    // Init the chart first.
    current_data_view = update_data_view();
    flush();

    change2FineTuned = function () {
        current_tune_flag = true;
        flush();
    };

    change2NotFineTuned = function () {
        current_tune_flag = false;
        flush();
    };

    dashboard.bind(filter, chart);
    dashboard.draw(data_table);
}