        google.charts.load('current', {'packages': ['corechart']});
        google.charts.setOnLoadCallback(drawChart);

        function drawChart() {
            var data = google.visualization.arrayToDataTable([
                ['Age', 'Weight'],
                [8, 12],
                [4, 5.5],
                [11, 14],
                [4, 5],
                [3, 3.5],
                [6.5, 7]
            ]);

            data.addColumn({type: 'string', role: 'tooltip'});

            var options = {
                title: 'Age vs. Weight comparison',
                hAxis: {title: 'X', minValue: 0, maxValue: 15},
                vAxis: {title: 'Y', minValue: 0, maxValue: 15},
                legend: 'none',
                // Allow multiple
                // simultaneous selections.
                selectionMode: 'multiple',
                // Trigger tooltips
                // on selections.
                tooltip: {trigger: 'selection'},
            };

            function placeMarker(dataTable) {

                var selectedItem = this.getSelection()[0];

                var cli = this.getChartLayoutInterface();
                // var chartArea = cli.getChartAreaBoundingBox();
                // "Zombies" is element #5.

                if (selectedItem) {
                    // console.log(dataTable.getRowProperties(selectedItem.row));
                    document.querySelector('.overlay-marker').style.top = Math.floor(cli.getYLocation(dataTable.getValue(selectedItem.row, selectedItem.column))) - 50 + "px";
                    // getValue(x, y) return the value in the cell of the table.
                    // So when you ask (x, 0), you return the cell in first column, which is the x-axis value.
                    document.querySelector('.overlay-marker').style.left = Math.floor(cli.getXLocation(dataTable.getValue(selectedItem.row, 0))) - 10 + "px";
                }
                else {
                    // This can remove the figure. But in the future I want to remove it from memory rather than making invisible.
                    document.querySelector('.overlay-marker').style.top = -1000;
                    document.querySelector('.overlay-marker').style.left = -1000;

                }
            };

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

            // Listen for the 'select' event, and call my function selectHandler() when
            // the user selects something on the chart.
            google.visualization.events.addListener(chart, 'select',
                placeMarker.bind(chart, data));
            // google.visualization.events.addListener(chart, 'select', selectHandler);


            chart.draw(data, options);
        }