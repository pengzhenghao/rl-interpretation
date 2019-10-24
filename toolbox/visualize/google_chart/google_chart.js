google.charts.load('current', {'packages': ['corechart']});
google.charts.setOnLoadCallback(drawChart);

var data_table;

var jsonData = $.ajax({
    url: 'test_data.json',
    dataType:"json",
    async: false
}).responseText;
// var jsonData = '{"cols":[{"id":"Age","type":"number"},{"id":"Weight","type":"number"},{"type":"string","role":"tooltip","p":{"html":true,"role":"tooltip"}}],"row":[{"c":[{"v":8},{"v":12},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":4},{"v":5.5},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":11},{"v":14},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":4},{"v":5},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":3},{"v":3.5},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":6.5},{"v":7},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]}]}';
// var jsonData = '{"cols":[{"id":"Age","type":"number"},{"id":"Weight","type":"number"},{"id":"Url","type":"string"}],"row":[{"c":[{"v":8},{"v":12},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":4},{"v":5.5},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":11},{"v":14},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":4},{"v":5},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":3},{"v":3.5},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]},{"c":[{"v":6.5},{"v":7},{"v":"https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg"}]}]}';

function drawChart() {
    // data_table = google.visualization.arrayToDataTable([
    //     ['Age', 'Weight', {
    //         'type': 'string',
    //         'role': 'tooltip',
    //         'p': {'html': true}
    //     }],
    //     [8, 12, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
    //         "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
    //     [4, 5.5, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
    //         "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
    //     [11, 14, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
    //         "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
    //     [4, 5, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
    //         "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
    //     [3, 3.5, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
    //         "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
    //     [6.5, 7, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
    //         "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')]
    // ]);
    // var json_str = $.getJSON('test_data.json');

    data_table = new google.visualization.DataTable(jsonData);


        // function placeMarker(dataTable) {
    //     var selectedItem = this.getSelection()[0];
    //     var cli = this.getChartLayoutInterface();
    //     // var chartArea = cli.getChartAreaBoundingBox();
    //
    //     if (selectedItem) {
    //
    //         for (var item in this.getSelection()){
    //
    //         // console.log(dataTable.getRowProperties(selectedItem.row));
    //         document.querySelector('.overlay-marker').style.top = Math.floor(cli.getYLocation(dataTable.getValue(selectedItem.row, selectedItem.column))) - 50 + "px";
    //         // getValue(x, y) return the value in the cell of the table.
    //         // So when you ask (x, 0), you return the cell in first column, which is the x-axis value.
    //         document.querySelector('.overlay-marker').style.left = Math.floor(cli.getXLocation(dataTable.getValue(selectedItem.row, 0))) - 10 + "px";
    //         }
    //
    //     } else {
    //         // This can remove the figure. But in the future I want to remove it from memory rather than making invisible.
    //         document.querySelector('.overlay-marker').style.top = -1000;
    //         document.querySelector('.overlay-marker').style.left = -1000;
    //     }
    // }

    var chart = new google.visualization.ScatterChart(document.getElementById('chart_div'));
    // The select handler. Call the chart's getSelection() method
    // function selectHandler() {
    //     var selectedItem = chart.getSelection()[0];
    //     if (selectedItem) {
    //         // console.log(chart.getSelection());
    //         // for (var newItem in chart.getSelection()){
    //         //     var value = data.getValue(newItem.row, newItem.column);
    //         var value = data.getValue(selectedItem.row, selectedItem.column);
    //         alert('The user selected ' + value);
    //         // }
    //     }
    // }
    //
    // for (var row=0; row<data_table.getNumberOfCoumns();row++){
    //     data_table.setCell(row, 2, data_table)
    // }



    function createCustomHTMLContent(hyperLink, imagePath, agentName) {
        return '<dev style="padding:5px 5px 5px 5px;"><a href="' + hyperLink + '"><img src=\"' +
            imagePath // here is image path
            + '" id="' + agentName +
            '" style=\"width:100px; height:100px\"  ></a></dev>';
    }

    var options = {
        tooltip: {isHtml: true, trigger: 'selection'},
        title: 'Age vs. Weight comparison',
        hAxis: {title: 'X', minValue: 0, maxValue: 15},
        vAxis: {title: 'Y', minValue: 0, maxValue: 15},
        legend: 'none',
        aggregationTarget: 'none',
        selectionMode: 'multiple'
    };

    // var chart = new google.visualization.ScatterChart(document.getElementById('chart_div'));

    chart.draw(data_table, options);
}