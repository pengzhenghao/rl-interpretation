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
        [8, 12, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
            "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
        [4, 5.5, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
            "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
        [11, 14, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
            "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
        [4, 5, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
            "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
        [3, 3.5, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
            "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')],
        [6.5, 7, createCustomHTMLContent('https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
            "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", 'China')]
    ]);

    function createCustomHTMLContent(hyperLink, imagePath, agentName) {
        return '<dev style="padding:5px 5px 5px 5px;"><a href="' + hyperLink + '"><img src=\"' +
            imagePath // here is image path
            + '" id="' + agentName +
            '" style=\"width:100px; height:100px\"  ></a></dev>';
        // return '<div style="padding:5px 5px 5px 5px;">' +
        //     '<img src="' + flagURL + '" style="width:75px;height:50px"><br/>' +
        // '</div>';
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