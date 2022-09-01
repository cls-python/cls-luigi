
async function fetchJSONRepo(path = "repo.json"){
  const fetched = await fetch(path);

  return fetched.json();
}


async function drawStaticRepo(path = "static_repo.json") {

  // Set up zoom support
  var svg = d3.select("svg.live-map"),
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



const repo = await fetchJSONRepo(path);

async function getTotalNumberOfPipelines(r){
  await r;

  let n = 1;
  for (var component in r){
    if (r[component]["abstract"]){
      n = n * r[component]["concreteImplementations"].length;
    }
  }
  return n;
}

d3.select("total-pipelines")
  .append("div")
  .classed("total-pipelines", true)
  .append("div")
  .classed("title", true)
  .text("Total Number of Pipelines: " + await getTotalNumberOfPipelines(repo));


async function AddNodesAndEdges(r, withLable=false) {
  await r;
  for (let component in r) {
    let componentDetails = r[component];

     // Add Node
    let html = "<div>";
    let className = "notAbstractComponent";
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

    // Add Edge
    if (componentDetails.inputQueue){
      for (const d of componentDetails.inputQueue){
        g.setEdge(d, component, {});
      }
    }
  }
}
await AddNodesAndEdges(repo, withLable=true);

inner.call(render, g);

// Zoom and scale to fit
var graphWidth =  g.graph().width + 80;
var graphHeight =  g.graph().height + 40;
var width =  parseInt(svg.style("width").replace(/px/, ""));
var height =  parseInt(svg.style("height").replace(/px/, ""));
var zoomScale =  Math.min(width / graphWidth, height / graphHeight);
var translateX =  (width / 2) - ((graphWidth * zoomScale) / 2) + 25;
var translateY =  (height / 2) - ((graphHeight * zoomScale) / 2) + 10;
svg.call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(zoomScale));
}


async function drawDynamicRepo(path="dynamic_repo.json") {


  // Set up zoom support
  var svg = d3.select("svg.detailed-repo"),
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

  const repo = await fetchJSONRepo(path);

  async function AddNodesAndEdges(r, withLable=false) {
    await r;
    for (let component in r) {
      let componentDetails = r[component];
  
       // Add Node
      let html = "<div>";
      let className = "notAbstractComponent";
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
  
      // Add Edge
      if (componentDetails.inputQueue){
        for (const d of componentDetails.inputQueue){
          g.setEdge(d, component, {});
        }
      }
    }
  }
  await AddNodesAndEdges(repo, withLable=true);
  
  inner.call(render, g);
  
  // Zoom and scale to fit
  var graphWidth =  g.graph().width + 80;
  var graphHeight =  g.graph().height + 40;
  var width =  parseInt(svg.style("width").replace(/px/, ""));
  var height =  parseInt(svg.style("height").replace(/px/, ""));
  var zoomScale =  Math.min(width / graphWidth, height / graphHeight);
  var translateX =  (width / 2) - ((graphWidth * zoomScale) / 2) + 25;
  var translateY =  (height / 2) - ((graphHeight * zoomScale) / 2) + 10;
  svg.call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(zoomScale));


}
document.addEventListener("DOMContentLoaded", drawStaticRepo());
document.addEventListener("DOMContentLoaded", drawDynamicRepo());

