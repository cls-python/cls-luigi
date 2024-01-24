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
        'border-width': '1.5px',
        'border-color': '#bbb',
      }
    },
    {
      selector: '.parent',
      style: {
        'background-color': '#aeaeae',
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
        'background-color': '#169f31',
        'shape': 'round-rectangle',
        'z-index': '10',
        'border-width': '1.5px',
        'border-color': '#bbb',

      }
    },

    {
      selector: '.abstract-task',
      style: {
        'label': 'data(label)',
        'background-color': '#11479e',
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
          'background-color': '#d5026f',
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
        'background-color': '#02d5cb',
        'shape': 'round-diamond',
        'z-index': '10',
        'border-width': '1.5px',
        'border-color': '#bbb',

    }
    },

    {
      "selector": ".outline",
      "style": {
        "color": "#fff",
        "text-outline-color": "#888",
        "text-outline-width": 3,
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
        'line-color': 'red',
        'target-arrow-color': 'red',
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
        'width': 10,
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
