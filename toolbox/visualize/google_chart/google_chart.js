google.charts.load('current', {'packages': ['corechart', 'controls']});
google.charts.setOnLoadCallback(drawChart);

var data_table;

var rawData = $.parseJSON($.ajax({
    url: 'test_data.json',
    dataType: "json",
    async: false
}).responseText);


function drawChart() {
    data_table = new google.visualization.DataTable(rawData['data']);

    // Fill the tooltip
    data_table.addColumn(
        {
            'type': 'string',
            'role': 'tooltip',
            'p': {'html': true, 'role': "tooltip"}
        }
    );

    function createCustomHTMLContent(hyperLink, imagePath, agentName) {
        return '<dev style="padding:5px 5px 5px 5px;"><a href="' + hyperLink + '"><img src=\"' +
            imagePath // here is image path
            + '" id="' + agentName +
            '" style=\"width:100px; height:100px\"  ></a></dev>';
    }

    var url, link, cell_html, row;
    for (row = 0; row < data_table.getNumberOfRows(); row++) {
        url = data_table.getRowProperty(row, "url");
        link = data_table.getRowProperty(row, "link");
        cell_html = createCustomHTMLContent(link, url, row);
        // console.log(cell_html);
        data_table.setCell(row, 2, cell_html);
    }

    // Create the chart
    var chart = new google.visualization.ScatterChart(document.getElementById('chart_div'));

    var info = rawData['info']
    var options = {
        tooltip: {isHtml: true, trigger: 'selection'},
        title: info['title'],
        hAxis: {
            title: info['xlabel'],
            minValue: info['xlim'] ? info['xlim'][0] : null,
            maxValue: info['xlim'] ? info['xlim'][1] : null
        },
        vAxis: {
            title: info['ylabel'],
            minValue: info['ylim'] ? info['ylim'][0] : null,
            maxValue: info['ylim'] ? info['ylim'][1] : null
        },
        legend: 'none',
        aggregationTarget: 'none',
        selectionMode: 'multiple'
    };

    chart.draw(data_table, options);

    // Create the slider for std changing
    var slider = document.getElementById("myRange");
    var output = document.getElementById("demo");
    output.innerHTML = slider.value;

    slider.oninput = function () {
        var std = get_exact_std(this.value)
        output.innerHTML = std;
        change_data(std);
    };

    function get_exact_std(slider_value) {
        return slider_value * 1.5
    }

    function change_data(std) {
        return null
    }

}