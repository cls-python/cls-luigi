
    /**
    * Helper to print out compound nodes and all child nodes.
    * Just for debug.
    * @param {CompoundNode} node - The compound node to explore and print out.
    */
    function printCompoundNodeAndChildren(node) {

        var position = node.position();
        var width = node.width();
        var height = node.height();

        // Access the compound node properties as needed
        console.log('Compound Node:', node.id());
        console.log('Node Position:', position);
        console.log('Node width:', width);
        console.log('Node height:', height);

        var positions = {};
        node.forEach(function (node) {
          positions[node.id()] = node.position();

          // Get all the child nodes of the parent node
          var childNodes = node.children();

          // Iterate over the child nodes
          childNodes.forEach(function (childNode) {
            // Access the child node properties as needed
            console.log('Child Node:', childNode.id());
            console.log('Child Node Position:', childNode.position());
            console.log('Child Node width:', childNode.width());
            console.log('Child Node height:', childNode.height());
          });
        });
      }

      function getMaxNodeSize(nodes){

        var maxWidth = 0;
        var maxHeight = 0;
        var padding= 15;

        nodes.forEach(function (node) {

            var label = node.data('label');
            if (label === undefined) {
            return;
            }
                // Create a temporary DOM element to measure the label size
                var tempElement = document.createElement('span');
                tempElement.style.visibility = 'hidden';
                tempElement.style.position = 'absolute';
                tempElement.style.display = 'inline-block';
                tempElement.style.padding = padding + 'px';
                tempElement.innerText = label;
                document.body.appendChild(tempElement);

                var labelWidth = tempElement.offsetWidth;
                var labelHeight = tempElement.offsetHeight;

                if (labelWidth > maxWidth){
                    maxWidth = labelWidth;
                };
                if (labelHeight > maxHeight){
                    maxHeight = labelHeight;
                };
                // Clean up the temporary DOM element
                document.body.removeChild(tempElement);

          });

          return {width: maxWidth, height: maxHeight, padding: padding};


      }

      /**
      * Helper to resize node to fit the attached label.
      * @param {Node} node - The node that should be resized.
      */
      function resizeNodeToFitLabel(node, maxNodeSize) {

        // Update the node size based on the label size
        node.style('width', maxNodeSize.width);
        node.style('height', maxNodeSize.height);
        node.style('padding', maxNodeSize.padding);

      };

      /**
      * Recursive function to traverse the compound node and its descendants.
      * Adds them to a dataStructure to keep track of them.
      * @param {CompoundNode} node - The compound node to travers.
      * @return - Fills the dataStructure(call by reference).
      */
      function traverseCompoundNode(node, dataStructure) {

        if (node.isNode()) {
          // For inner nodes, retrieve the data and add it to the data structure
          var nodeData = node.data();
          var nodeId = nodeData.id;
          var nodeStyle = node.style();
          var nodeClasses = node.classes();
          if (!dataStructure.innerNodes.hasOwnProperty(nodeId)) {
            // Adding the new item to the dictionary
            dataStructure.innerNodes[nodeId] = { data: nodeData, style: nodeStyle, classes: nodeClasses, dimensions: {} };
          }

          var connectedEdges = node.connectedEdges();

          // Loop through the connected edges and perform desired operations
          connectedEdges.forEach(function (edge) {
            var edgeData = edge.data();
            var edgeId = edgeData.id;
            var edgeStyle = edge.style();
            var edgeClasses = edge.classes();
            if (!dataStructure.innerEdges.hasOwnProperty(edgeId)) {
              // Adding the new item to the dictionary
              dataStructure.innerEdges[edgeId] = { data: edgeData, style: edgeStyle, classes: edgeClasses };

            }

          });

        } else if (node.isEdge()) {
          // For inner edges, retrieve the data and add it to the data structure
          var edgeData = edge.data();
          var edgeId = edgeData.id;
          var edgeStyle = edge.style();
          var edgeClasses = edge.classes();
          if (!dataStructure.edges.hasOwnProperty(edgeId)) {
            // Adding the new item to the dictionary
            dataStructure.edges[edgeId] = { data: edgeData, style: edgeStyle, classes: edgeClasses };
          }
        }
        // Traverse the children of the compound node recursively
        node.children().forEach(function (childNode) {
          traverseCompoundNode(childNode, dataStructure);
        });
      };


      function loadScript(url, callback) {
        var script = document.createElement('script');
        script.type = 'text/javascript';

        // Check if the script has successfully loaded
        script.onload = function() {
          callback();
        };

        // If the script fails to load
        script.onerror = function() {
          console.error('Failed to load ' + url);
        };

        script.src = url;
        document.head.appendChild(script);
      };
