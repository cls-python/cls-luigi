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
};

/**
 * Calculates the maximum width and height of the labels among a collection of nodes.
 *
 * @param {Array} nodes - An array of node objects containing 'label' data.
 * @returns {Object} - An object containing the maximum width and height of labels along with padding.
 * @description This function iterates through the provided array of node objects to calculate the
 * maximum width and height of the labels. It creates temporary DOM elements to measure the size of
 * each label and considers padding to ensure proper spacing. The function returns an object containing
 * the maximum width and height of labels along with the padding used.
 */
function getMaxNodeSize(nodes) {

    var maxWidth = 0;
    var maxHeight = 0;
    var padding = 15;

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

        if (labelWidth > maxWidth) {
            maxWidth = labelWidth;
        };
        if (labelHeight > maxHeight) {
            maxHeight = labelHeight;
        };
        // Clean up the temporary DOM element
        document.body.removeChild(tempElement);

    });

    return { width: maxWidth, height: maxHeight, padding: padding };


};

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
    script.onload = function () {
        callback();
    };

    // If the script fails to load
    script.onerror = function () {
        console.error('Failed to load ' + url);
    };

    script.src = url;
    document.head.appendChild(script);
};

/**
 * Returns a collection of compound nodes from a given Cytoscape.js graph instance.
 *
 * @param {object} cy - The Cytoscape.js graph instance.
 * @returns {object} - A collection of compound nodes.
 * @description This function iterates through all nodes in the Cytoscape.js graph
 * instance and filters out only the compound nodes. Compound nodes are nodes that
 * contain child nodes (also referred to as "parent" nodes). The function returns
 * a collection of these compound nodes.
 */
function getCompoundNodesOfGraph(cy) {
    return cy.nodes().filter(function (node) {
        return node.isParent();
    });
};

/**
 * Filters out inheritance edges that do not connect inner nodes within a compound node's data object.
 *
 * @param {object} compoundNodeData - An object containing inner nodes and edges data of a compound node.
 * @returns {undefined}
 * @description This function iterates over the inner edges of a compound node represented by the
 * provided 'compoundNodeData' object. It filters out inheritance edges that do not connect inner nodes
 * and updates the 'compoundNodeData' object accordingly by removing these edges from the inner edges data.
 * The function modifies the 'compoundNodeData' object in place.
*/
function filterForInheritanceEdges(compoundNodeData) {
    Object.keys(compoundNodeData.innerEdges).forEach(function (edgeId) {
        var edge = compoundNodeData.innerEdges[edgeId];

        if (!compoundNodeData.innerNodes.hasOwnProperty(edge.data.target) || !compoundNodeData.innerNodes.hasOwnProperty(edge.data.source)) {
            var edgeData = { id: edge.data.id, source: edge.data.source, target: edge.data.target };
            var edgeClasses = edge.classes;
            compoundNodeData.filteredEdges[edgeId] = { data: edgeData, classes: edgeClasses };
        }
    });

    Object.keys(compoundNodeData.filteredEdges).forEach(function (id) {

        delete compoundNodeData.innerEdges[id];

    });
};

/**
 * Perform manual level wrap layouting a compoundNode subgraph.
 *
 * @param {Array} outgoingEdges - An array containing outgoing edge data.
 * @param {object} subgraphCy - The Cytoscape.js instance representing the subgraph.
 * @returns {undefined}
 * @description This function performs manual level wrap layouting. It divides the
 * outgoing edges into chunks based on a predefined maximum number of nodes on each
 * level, and then updates the target of edges in each subsequent chunk to maintain
 * a hierarchical layout. The function modifies the subgraph Cytoscape.js instance in place.
 */
function doManualLevelWrapLayouting(outgoingEdges, subgraphCy) {

    /**
    * seperate outgoingEdges list into list of lists of maxNodesInOnLevel
    * put list[x+1] as childs of middle node in list[x]
    * update edges
    */

    var edgeChunks = [];

    for (var i = 0; i < outgoingEdges.length; i += maxNodesInOnLevel) {
        var chunk = outgoingEdges.slice(i, i + maxNodesInOnLevel);
        edgeChunks.push(chunk);
    }

    // Iterate over the list of lists and update the target of edges in list x+1
    for (var j = 0; j < edgeChunks.length - 1; j++) {
        var currentChunk = edgeChunks[j];
        var nextChunk = edgeChunks[j + 1];

        // Get the middle node in list x
        var middleNodeIndex = Math.floor(currentChunk.length / 2);
        var middleNodeId = currentChunk[middleNodeIndex].data().target;

        // Iterate over edges in list x+1
        for (var k = 0; k < nextChunk.length; k++) {
            // Update the target of the edge in list x+1
            var edgeId = nextChunk[k].data().id;
            var edge = subgraphCy.getElementById(edgeId);
            var edgeData = edge.data();
            var edgeClasses = edge.classes;

            var newSource = middleNodeId;
            var newTarget = edgeData.target;

            edge.remove();

            subgraphCy.add({
                group: 'edges',
                data: { id: edgeId, source: newSource, target: newTarget },
                classes: edgeClasses
            });

        }
    }
};

function checkFileChange(lastModified, callback) {
    var xhr = new XMLHttpRequest();
    xhr.open('HEAD', "graphdata.js");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                var newLastModified = xhr.getResponseHeader('Last-Modified');
                if (newLastModified !== lastModified.time) {
                    lastModified.time = newLastModified;
                    callback();
                }
            }
        }
    };
    xhr.send();
}

var fileChangedFunction = function () {
    console.log('File changed, reload graph file!');
    createCytoscapeStaticGraph();
};
