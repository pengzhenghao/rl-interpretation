var data;
var agent_name_list;
var num_cols = 10;
var num_rows = 0;
var max_agents = 100;
var path_prefix = "";
var default_mode = "clip";
var display_info_keys = ['performance', 'distance'];
var numerical_info_keys = ['performance', 'distance'];

// var json_path = "index.json";
var cluster_json_path = "cluster.json";

var cluster_list;
var agent_cluster_mapping;
var json_path;

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
    var cnt = 0;
    for (x in data) {
        agent_name_list.push(x);
        cnt += 1;
        if (cnt === max_agents) {
            break;
        }
    }
    return agent_name_list;
};

function collectAgentInfo(agent_data, cluster_info) {
    var ret = "";
    var info_dict = agent_data['info'];
    var info_key;
    for (info_key in info_dict) {
        if ($.inArray(info_key, display_info_keys) >= 0) {
            var val = info_dict[info_key];
            if ($.inArray(info_key, numerical_info_keys) >= 0) {
                val = Number(val).toFixed(2);
            }
            ret += info_key + ": " + val + "<br>";
        }
    }
    for (info_key in cluster_info) {
        if ($.inArray(info_key, display_info_keys) >= 0) {
            var val = info_dict[info_key];
            if ($.inArray(info_key, numerical_info_keys) >= 0) {
                val = Number(val).toFixed(2);
            }
            ret += info_key + ": " + val + "<br>";
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

function fillClusterData(mode) {
    var agent_name;
    // var cluster;
    var row = 0;
    var num_cluster = cluster_list.length;
    var cols = Array();

    for (agent_name of agent_cluster_mapping) {
        var cluster_info = agent_cluster_mapping[agent_name];
        var cluster = cluster_info['cluster'];
        var col = cluster_info['col'];
        var ceil_id = getCeilId(cluster, col);

        var head = document.getElementById(ceil_id + "head");
        head.innerHTML = agent_name;

        var img = document.getElementById(ceil_id + "img");
        var agent_data = data[agent_name];
        var src_path = agent_data['gif_path'][mode] + "?a=" + Math.random();
        var href = agent_data['gif_path']['hd'] + "?a=" + Math.random();

        img.innerHTML = "<a href=\"" + href + "\"><img src=\""
            + path_prefix + src_path + "\" id=\"" + agent_name +
            "\" style=\"width:100px; height:100px\"  ></a>";

        // "<a href=" + +  ">     <img src="flower.jpg"
        // style="width:82px; height:86px" title="White flower" alt="Flower">   </a>

        var info = document.getElementById(ceil_id + "info");
        // info.innerHTML = agent_name + "info";
        info.innerHTML = collectAgentInfo(agent_data, cluster_info);

        col += 1;
        if (col === num_cols) {
            row += 1;
            col = 0;
        }
    }
}


function onclick_clip() {
    fillClusterData('clip');
}

function onclick_1period() {
    fillClusterData("period");
}

function onclick_3period() {
    fillClusterData("3period");
}

function onclick_beginning() {
    fillClusterData("beginning");
}

function onclick_end() {
    fillClusterData("end");
}

function read_index_json() {
    $.getJSON(json_path, function (read_data) {
        data = read_data;
        agent_name_list = get_agent_name_list(data);
        console.log("At the beginning, agent list is: " + agent_name_list);
        createTable();
        // fillDataDefault(default_mode);
    });
}


$.getJSON(cluster_json_path, function (read_data) {
    cluster_list = read_data['cluster_list'];
    agent_cluster_mapping = read_data['agent_cluster_mapping'];
    agent_name_list = get_agent_name_list(agent_cluster_mapping);
    json_path = read_data['json_path'];

    read_index_json();
    fillClusterData(default_mode);

});