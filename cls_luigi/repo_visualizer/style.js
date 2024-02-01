var dagre_tb_layout = {
    name: 'dagre',
    nodeSep: 80,
    edgeSep: 20,
    rankSep: 80,
    rankDir: 'TB',
    fit: true,
    padding: 50,
    stop: true,
    minLen: 2,



  };

  var dagre_lr_layout = {
    name: 'dagre',
    nodeSep: 100,
    edgeSep: 30,
    rankSep: 80,
    rankDir: 'LR',
    fit: true,
    padding: 50,
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
        'shape': 'round-diamond',
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
        'width': 8,
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
        'width': 8,
        'source-arrow-shape': 'triangle-backcurve',
        'line-color': 'black',
        'target-arrow-shape': 'none',
        'line-style': 'solid',
        'source-arrow-color': 'black',
        'curve-style': 'taxi',
        'taxi-direction': 'downward',
        'z-index': '6',
      }
    },
  ];
