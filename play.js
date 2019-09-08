var data;
var agent_name_list;
var num_cols = 2;
var num_rows = 0;
var path_prefix = "";
var default_mode = "clip";
var display_info_keys = ['performance'];

// The format of json file:
// {
//     "agent 1": {
//         "column": 0,
//         "row": 1,
//         "name": "agent 1",
//         "gif_path": {
//             "end": "gif/test-2-agents/3period/sfsad 1.gif",
//             "clip": ...
//         },
//         "info": {
//             "performance": 100,
//             "length": 101,
//             "xxx": ...,
//         }
//     },
//     ...
// }


var get_agent_name_list = function (data) {
    var agent_name_list = [];
    var x;
    for (x in data) {
        agent_name_list.push(x);
    }
    return agent_name_list;
};

function collectAgentInfo(agent_data) {
    var ret = "";
    var info_dict = agent_data['info'];
    for (var info_key in info_dict) {
        if ($.inArray(info_key, display_info_keys)>=0) {
            ret += info_key + ": " + info_dict[info_key] + "<br>";
        }
    }
    return ret;
}

function getCeilId(row, col) {
    return "(" + row + "," + col + ")";
}

function createTable() {
    var num_agents = agent_name_list.length;
    num_rows = Math.ceil(num_agents / num_cols);

    var row;
    var col;

    // build the framework first.
    var tbody = document.getElementById("tbody");
    for (row = 0; row < num_rows; row++) {

        var tr_head = document.createElement("tr");
        var tr_img = document.createElement("tr");
        var tr_info = document.createElement("tr");

        for (col = 0; col < num_cols; col++) {
            ceil_id = getCeilId(row, col);
            var th = document.createElement("th");
            th.id = ceil_id + "head";
            tr_head.appendChild(th);

            var img = document.createElement("td");
            img.id = ceil_id + "img";
            tr_img.appendChild(img);

            var info = document.createElement("td");
            info.id = ceil_id + "info";
            tr_info.appendChild(info);
        }
        var table = document.createElement("table");
        table.appendChild(tr_head);
        table.appendChild(tr_img);
        table.appendChild(tr_info);
        tbody.appendChild(table);
    }
}

function fillDataDefault(mode) {
    var agent_name;
    var row = 0;
    var col = 0;
    for (agent_name of agent_name_list) {
        var ceil_id = getCeilId(row, col);

        var head = document.getElementById(ceil_id + "head");
        head.innerHTML = agent_name;

        var img = document.getElementById(ceil_id + "img");
        var agent_data = data[agent_name];
        var src_path = agent_data['gif_path'][mode];
        img.innerHTML = "<img src=\""
            + path_prefix + src_path + "\" id=\"" + agent_name +
            "\" />";

        var info = document.getElementById(ceil_id + "info");
        // info.innerHTML = agent_name + "info";
        info.innerHTML = collectAgentInfo(agent_data);

        col += 1;
        if (col === num_cols) {
            row += 1;
            col = 0;
        }
    }
}

function onclick_clip() {
    fillDataDefault('clip');
}

function onclick_period() {
    fillDataDefault("3period");
}

function onclick_beginning() {
    fillDataDefault("beginning");
}

function onclick_end() {
    fillDataDefault("end");
}

$.getJSON("test-html.json", function (read_data) {
    data = read_data;
    agent_name_list = get_agent_name_list(data);
    console.log("At the beginning, agent list is: " + agent_name_list);
    createTable();
    fillDataDefault(default_mode);
});