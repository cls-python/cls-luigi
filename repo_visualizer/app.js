async function fetchJSONRepo(path = "repo.json") {
  const response = await fetch(path);

  return response.json();
};

const repo = await fetchRepo();



  // Set up zoom support
var svg = d3.select("svg"),
  inner = svg.append("g"),
  zoom = d3.zoom().on("zoom", function() {
    inner.attr("transform", d3.event.transform);
  });
  svg.call(zoom);

var render = new dagreD3.render();

  // Left-to-right layout
var g = new dagreD3.graphlib.Graph()
  .setGraph({
  nodesep: 100,
  ranksep: 70,
  rankdir: "LR"
});
    
function addNode(component){
  var componentDetails = repo[component];
    var html = "<div>";
    var className = "notAbstractComponent";
    html += "<span class=status></span>";
    html += "<span class=name>"+component+"</span>";
    html += "<span class=queue>" + ""+"</span>";


    if (componentDetails.concreteImplementations){
      className = "abstractComponent"
      for (const i of componentDetails.concreteImplementations){
        html += "<span class=queue>"+i+"</span>";
      }
      html += "<span class=queue>"+""+"</span>";
    }
    html += "</div>";

    g.setNode(component , {
      labelType: "html",
      label: html,
      rx: 5,
      ry: 5,
      padding: 0,
      class: className,
    });
};

function addEdge(component, withLable= true) {
  var componentDetails = repo[component];
  if (componentDetails.inputQueue){
    for (const d of componentDetails.inputQueue){
      var edgeLable = NaN;
      if (repo[d]["abstract"]){
        edgeLable = repo[d]["concreteImplementations"].length.toString();
      }
      if (withLable){
        g.setEdge(d, component, {label: edgeLable});
      } else {
        g.setEdge(d, component, {});
      }
    }
  }
};

function draw(isUpdate) {
  for (var component in repo) {
    addNode(component);
    addEdge(component, withLable=false);
  }
  inner.call(render, g);
  
  // Zoom and scale to fit
  var graphWidth = g.graph().width + 80;
  var graphHeight = g.graph().height + 40;
  var width = parseInt(svg.style("width").replace(/px/, ""));
  var height = parseInt(svg.style("height").replace(/px/, ""));
  var zoomScale = Math.min(width / graphWidth, height / graphHeight);
  var translateX = (width / 2) - ((graphWidth * zoomScale) / 2) + 25;
  var translateY = (height / 2) - ((graphHeight * zoomScale) / 2);
  var svgZoom = isUpdate ? svg.transition().duration(500) : svg;
  svgZoom.call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(zoomScale));
}


function getTotalNumberOfPipelines(){
  var n = 1;
  for (var component in repo){
    if (repo[component]["abstract"] === true){
      n = n * repo[component]["concreteImplementations"].length;
    }
  }
  return n;
}


d3.select("total-pipelines")
  .append("div")
  .classed("total-piplelines", true)
  .append("div")
  .classed("title", true)
  .text("Total Number of Pipelines: " + getTotalNumberOfPipelines());





document.addEventListener("DOMContentLoaded", draw);