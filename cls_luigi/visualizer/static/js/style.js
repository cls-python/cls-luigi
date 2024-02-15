/*
#
# Apache Software License 2.0
#
# Copyright (c) 2022-2023, Jan Bessai, Anne Meyer, Hadi Kutabi, Daniel Scholtyssek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
*/

var dagre_tb_layout = {
    name: 'dagre',
    nodeSep: 40,
    rankSep: 40,
    rankDir: 'TB',
    fit: true,
    stop: true,
  };

  var dagre_lr_layout = {
    name: 'dagre',
    nodeSep: 60,
    edgeSep: 100,
    rankSep: 60,
    rankDir: 'LR',
    fit: true,
    padding: 30,
    minLen: 2,
    stop: true,

  };

  var globalStyle = [
    {
      selector: 'node',
      style: {
        'background-color': '#11479e',
        'z-index': '10',
      }
    },

    {
      selector: '.parent',
      style: {
        'background-color': '#7a7a7a',
        'text-valign': 'top',
        'text-halign': 'center',
        'shape': 'barrel',
        'z-index': '5',
        'border-width': '1.5px',
        'border-color': '#bbb',

      }
    },


    {
      selector: '.luigi-task',
      style: {
        'label': 'data(label)',
        'background-color': '#58814b',
        'shape': 'round-rectangle',
        'z-index': '10',
        'border-width': '1.5px',
        'border-color': '#bbb',
        "font-weight": "bold",

      }
    },

    {
      selector: '.abstract-task',
      style: {
        'label': 'data(label)',
        'background-color': '#8B0000',
        'shape': 'bottom-round-rectangle',
        'z-index': '10',
        'border-width': '1.5px',
        'border-color': '#bbb',

      }
    },

    {
        selector: '.concrete-task',
        style: {
          'label': 'data(label)',
          'background-color': '#3498db',
          'shape': 'round-rectangle',
          'z-index': '10',
          'border-width': '1.5px',
          'border-color': '#bbb',

        }
      },

    {
    selector: '.config-domain-task',
    style: {
        'label': 'data(label)',
        'background-color': '#001F3F',
        'shape': 'round-rectangle',
        'z-index': '10',
        'border-width': '1.5px',
        'border-color': '#bbb',

    }
    },

    {
      "selector": ".outline",
      "style": {
        "color": "#FFFFFF",
        "text-outline-color": "#FFFFFF",
        "text-outline-width": '0.5',
        "text-wrap": "wrap",
        'text-valign': 'center', // Center the text vertically
        'text-halign': 'center', // Center the text horizontally

      }
    },

    {
      selector: 'edge',
      style: {
        'width': 4,
        'line-style': 'dashed',
        'target-arrow-shape': 'vee',
        'line-color': '#cfcfcf',
        'target-arrow-color': '#cfcfcf',
        "curve-style": "bezier",
        "control-point-step-size": 40,
        //'curve-style' : 'taxi',
        //'taxi-direction' : 'rightward',
        'z-index': '7',
      }
    },
    {
      selector: '.inheritance',
      style: {
        'width': 4,
        'source-arrow-shape': 'triangle-backcurve',
        'line-color': 'black',
        'target-arrow-shape': 'none',
        'line-style': 'solid',
        'source-arrow-color': 'white',
        'curve-style': 'taxi',
        'taxi-direction': 'downward',
        'z-index': '6',
      }
    },
  ];
