function createCytoscapeStaticGraph() {

  loadScript("graphdata.js", function () {

    var cy = window.cy = cytoscape({
      container: document.getElementById('cy'),

      boxSelectionEnabled: false,
      autounselectify: true,
      wheelSensitivity: 0.4,
      style: globalStyle,
      elements: elementsData
    });


    /**
    * This subgraph is used to layout the compound nodes with a different layout.
    */
    var subgraphCy = cytoscape({
      container: document.getElementById('subgraph-container'),
      style: globalStyle,
    });

    // search for biggest node based on label, and fit every other node to same size.

    maxNodeSize = getMaxNodeSize(cy.nodes())

    // Layout of generated graph. Maybe not needed!?
    cy.nodes().forEach(function (node) {

      resizeNodeToFitLabel(node, maxNodeSize);

    });

    cy.layout(dagre_lr_layout).run();

    // Get all the compound nodes
    var compoundNodes = cy.nodes().filter(function (node) {
      return node.isParent();
    });


    /**
    * This keeps track of changed Compound Node data.
    * since we need to filter out everything expect inner nodes and edges
    * to draw them in the subgraph for layouting.
    * Maps CompoundNodeId to CompoundNodeData.
    */
    var changedCompoundNodeData = {};

    // Maps inner nodes to compound nodes (parent)
    var innerToCompoundMap = {};

    // Iterate over the compound nodes
    compoundNodes.forEach(function (node) {
      // Create an empty data structure to store the inner nodes and edges
      var compoundNodeData = {
        innerNodes: {},
        innerEdges: {},
        filteredEdges: {},
        targetSize: {},
        targetPosition: {}
      };

      changedCompoundNodeData[node.data().id] = compoundNodeData;

      // Start the traversal with the compound node
      traverseCompoundNode(node, compoundNodeData);

      console.log(compoundNodeData.innerNodes)

      // Fills innerToCompoundMap with content. Uses the current compoundNodeData.
      // Should be no problem since a node can only have one Parent.
      Object.keys(compoundNodeData.innerNodes).forEach(function (innerNodeId) {
        var innerNode = compoundNodeData.innerNodes[innerNodeId];


        if (innerNode.data.hasOwnProperty('parent')) {

          if (!(innerNode.data.id in innerToCompoundMap)) {
            innerToCompoundMap[innerNode.data.id] = innerNode.data.parent;
          }
        }

      });

      console.log("innerToCompoundMap", innerToCompoundMap);

      /**
      * Filter out edges that are not relevant for subgraph.
      * Namely every edge that goes in or out of the compound node.
      * Only edges left are "inheritance" edges.
      */
      Object.keys(compoundNodeData.innerEdges).forEach(function (edgeId) {
        var edge = compoundNodeData.innerEdges[edgeId];

        if (!compoundNodeData.innerNodes.hasOwnProperty(edge.data.target) || !compoundNodeData.innerNodes.hasOwnProperty(edge.data.source)) {
          var edgeData = { id: edge.data.id, source: edge.data.source, target: edge.data.target };
          var edgeClasses = edge.classes;
          compoundNodeData.filteredEdges[edgeId] = { data: edgeData, classes: edgeClasses };
        }
      });
      Object.keys(compoundNodeData.filteredEdges).forEach(function (id) {

        delete compoundNodeData.innerEdges[id]

      });
      console.log("filtertedEdge:", compoundNodeData.filteredEdges);
      console.log("innerNodes:", compoundNodeData.innerNodes);


      /**
      * Iterate over the innerNodes in dataStructure and adds them to the subgraph.
      */
      Object.keys(compoundNodeData.innerNodes).forEach(function (nodeId) {
        var node = compoundNodeData.innerNodes[nodeId];
        // Add the node to the Cytoscape instance

        if (subgraphCy.$id(nodeId).nonempty()) {
          console.log('The node exists in the graph');
        } else {
          subgraphCy.add({
            group: 'nodes',
            data: node.data,
            classes: node.classes
          });
        }
      });
      // resizes nodes to fit label. Will result in same size as in main graph.
      subgraphCy.nodes().forEach(function (node) {

        resizeNodeToFitLabel(node, maxNodeSize);

      });

      /**
      * Iterate over innerEdges in dataStructure and adds them to the subgraph.
      */
      Object.keys(compoundNodeData.innerEdges).forEach(function (edgeId) {
        var edge = compoundNodeData.innerEdges[edgeId];
        if (subgraphCy.$id(edgeId).nonempty()) {
          console.log('The edge exists in the graph');
        } else {
          // Add the edge to the Cytoscape instance
          subgraphCy.add({
            group: 'edges',
            data: edge.data,
            classes: edge.classes
          });
        }
      });
      // run Top-Bottom Layout for a classdiagram-like layout.
      subgraphCy.layout(dagre_tb_layout).run();

      var subCompoundNode = subgraphCy.getElementById(node.data().id);

      // Get the size (height and width) of the compound node with the new layout.
      var subCompoundNodeHeight = subCompoundNode.height();
      var subCompoundNodeWidth = subCompoundNode.width();

      // save it in compoundNodeData
      compoundNodeData.targetSize = { width: subCompoundNodeWidth, height: subCompoundNodeHeight };

      var childNodes = subCompoundNode.children();

      // Iterate over the child nodes and get width, height, position and save it to compoundNodeData.
      childNodes.forEach(function (childNode) {
        childWidth = childNode.width();
        childHeight = childNode.height();
        childX = childNode.position().x - subCompoundNode.position().x;
        childY = childNode.position().y - subCompoundNode.position().y;

        compoundNodeData.innerNodes[childNode.id()].dimensions = { width: childWidth, height: childHeight, x: childX, y: childY };


      });
    });

    // data structure to hold edges that have been changed in the layout graph.
    changedEdges = {}

    console.log("changedCompoundNodeData:", changedCompoundNodeData)
    Object.keys(changedCompoundNodeData).forEach(function (compoundNodeId) {

      var cNode = cy.getElementById(compoundNodeId);
      Object.keys(changedCompoundNodeData[compoundNodeId].filteredEdges).forEach(function (edgeId) {
        var edge = changedCompoundNodeData[compoundNodeId].filteredEdges[edgeId];
        var edgeData = edge.data;
        var edgeClasses = edge.classes;
        var sourceNode = cy.getElementById(edgeData.source);
        var targetNode = cy.getElementById(edgeData.target);

        // Remove the existing edge from the graph
        changedEdges[edgeId] = { source: edgeData.source, target: edgeData.target, classes: edgeClasses };
        var graphEdge = cy.getElementById(edgeId);
        graphEdge.remove();

        var newDataId = edgeData.id;
        var newSource = edgeData.source;
        var newTarget = edgeData.target;

        if (newSource in innerToCompoundMap) {
          newSource = innerToCompoundMap[edgeData.source];
        };

        if (newTarget in innerToCompoundMap) {
          newTarget = innerToCompoundMap[edgeData.target];
        };

        /**
        * @todo Error in Creation of Layout Graph.
        */

        cy.add({
          group: 'edges',
          data: { id: newDataId, source: newSource, target: newTarget },
          classes: edgeClasses
        });
      });


      var nodesToRemove = cNode.descendants().nodes();

      cy.remove(nodesToRemove)

      cNode.style("width", changedCompoundNodeData[compoundNodeId].targetSize.width);
      cNode.style("height", changedCompoundNodeData[compoundNodeId].targetSize.height);
      cNode.data('isParent', true);
    });

    cy.layout(dagre_lr_layout).run();

    // from here on out the layout is what we want. Empty Parent Nodes right now.
    //#1 for all compound nodes, saves position and current size.

    Object.keys(changedCompoundNodeData).forEach(function (compoundNodeId) {

      var cNode = cy.getElementById(compoundNodeId);
      var compoundNodeData = changedCompoundNodeData[compoundNodeId];

      var position_x = cNode.position().x;
      var position_y = cNode.position().y;

      compoundNodeData.targetPosition = { x: position_x, y: position_y };

      //#2 get all missing inner nodes of the current compound node and add them back,
      //   use the size  of subgraph nodes and the relative position of the subgraph
      //   nodes (x: x.child - x.parent, y: y.child - y.parent).

      Object.keys(compoundNodeData.innerNodes).forEach(function (nodeId) {
        if (!(nodeId === compoundNodeId)) {
          var node = compoundNodeData.innerNodes[nodeId];
          // Add the node to the Cytoscape instance
          cy.add({
            group: 'nodes',
            data: node.data,
            classes: node.classes,
            style: {
              width: node.dimensions.width,
              height: node.dimensions.height,
            },
            position: { x: node.dimensions.x + position_x, y: node.dimensions.y + position_y }
          });

        }
      });

      //#3 add all missing inner edges with the needed styling.
      Object.keys(compoundNodeData.innerEdges).forEach(function (edgeId) {
        var edge = compoundNodeData.innerEdges[edgeId];

        cy.add({
          group: 'edges',
          data: edge.data,
          classes: edge.classes
        });
      });
    });

    //#4 for all changed edges. get the current active edge (via ID) and change the source and target
    //   to the changed edges values. This should update the edges to the old ones.
    Object.keys(changedEdges).forEach(function (edgeId) {
      var edge = changedEdges[edgeId];

      var graphEdge = cy.getElementById(edgeId);
      graphEdge.remove();

      cy.add({
        group: 'edges',
        data: {
          id: edgeId,
          source: edge.source,
          target: edge.target,
        },
        classes: edge.classes
      });

    });

  });
};
