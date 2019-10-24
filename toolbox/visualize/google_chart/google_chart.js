google.charts.load('current', {'packages': ['corechart']});
google.charts.setOnLoadCallback(drawChart);

var data_table;

var jsonData = $.ajax({
    url: 'test_data.json',
    dataType: "json",
    async: false
}).responseText;

function drawChart() {


    data_table = new google.visualization.DataTable(jsonData);

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

    var options = {
        tooltip: {isHtml: true, trigger: 'selection'},
        title: 'Age vs. Weight comparison',
        hAxis: {title: 'X'},
        vAxis: {title: 'Y'},
        legend: 'none',
        aggregationTarget: 'none',
        selectionMode: 'multiple'
    };
    chart.draw(data_table, options);
}