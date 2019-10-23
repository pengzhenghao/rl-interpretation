google.charts.load('current', {'packages': ['corechart']});
google.charts.setOnLoadCallback(drawChart);

var data_table;

function drawChart() {
    data_table = google.visualization.arrayToDataTable([
        ['Age', 'Weight', {
            'type': 'string',
            'role': 'tooltip',
            'p': {'html': true}
        }],
        [8, 12, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg')],
        [4, 5.5, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg')],
        [11, 14, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg')],
        [4, 5, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg')],
        [3, 3.5, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg')],
        [6.5, 7, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg')]
    ]);

    function createCustomHTMLContent(flagURL) {
        return '<div style="padding:5px 5px 5px 5px;">' +
            '<img src="' + flagURL + '" style="width:75px;height:50px"><br/>' +
            // '<table class="medals_layout">' + '<tr>' +
            // '<td><img src="https://upload.wikimedia.org/wikipedia/commons/1/15/Gold_medal.svg" style="width:25px;height:25px"/></td>' +
            // '<td><b>' + totalGold + '</b></td>' + '</tr>' + '<tr>' +
            // '<td><img src="https://upload.wikimedia.org/wikipedia/commons/0/03/Silver_medal.svg" style="width:25px;height:25px"/></td>' +
            // '<td><b>' + totalSilver + '</b></td>' + '</tr>' + '<tr>' +
            // '<td><img src="https://upload.wikimedia.org/wikipedia/commons/5/52/Bronze_medal.svg" style="width:25px;height:25px"/></td>' +
            // '<td><b>' + totalBronze + '</b></td>' + '</tr>' + '</table>' +
        '</div>';
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

    var chart = new google.visualization.ScatterChart(document.getElementById('chart_div'));

    chart.draw(data_table, options);
}