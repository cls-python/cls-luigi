async function fetchJSON(path){
  const fetched = await fetch(path);

  return fetched.json();
}

async function addNodesAndEdges(jsonRepo, graph, graphType) {
  await jsonRepo;

  for (let component in jsonRepo) {
    let componentDetails = jsonRepo[component];
    let html = "<div>";
    let className;


    if (graphType === "static"){
      html += "<span class=status></span>";
      html += "<span class=name>" + component + "</span>";
      html += "<span class=queue>" + "" + "</span>";

      if (componentDetails.concreteImplementations) {
        className = "abstractComponent";
        for (let i of componentDetails.concreteImplementations) {
          html += "<span class=queue>" + i + "</span>";
        }
        html += "<span class=queue>" + "" + "</span>";
      }
      else if (componentDetails.configIndexes){
        className = "indexed";
        for (let i of componentDetails.configIndexes) {
          html += "<span class=queue>" + i + "</span>";
        }
        html += "<span class=queue>" + "" + "</span>";
      }
      else {

        className = "notAbstractComponent";
      }
    }

    else if (graphType === "dynamic") {
      className = componentDetails["status"];
      html += "<span class=status></span>";
      html += "<span class=name>"+component+"</span>";
      html += "<span class=queue>"+""+"</span>";
    }

    html += "</div>";

    graph.setNode(component, {
      labelType: "html",
      label: html,
      rx: 5,
      ry: 5,
      padding: 0,
      class: className,
      id: component
    });

    if (componentDetails.inputQueue) {
      for (let d of componentDetails.inputQueue) {
        graph.setEdge(d, component, {});
      }
    }
  }
}

async function draw(repo, g, svg, zoom, inner, render, dynamic=false){

  await addNodesAndEdges(repo, g,  "static");

  inner.call(render, g);

  // Zoom and scale to fit
  let graphWidth =  g.graph().width + 80;
  let graphHeight =  g.graph().height + 150;
  let width =  parseInt(svg.style("width").replace(/px/, ""));
  let height =  parseInt(svg.style("height").replace(/px/, ""));
  let zoomScale =  Math.min(width / graphWidth, height / graphHeight);
  let translateX =  (width / 2) - ((graphWidth * zoomScale) / 2) + 25;
  let translateY =  (height / 2) - ((graphHeight * zoomScale) / 2) + 10;
  svg.call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(zoomScale));

}


async function staticGraph(path = "static_repo.json") {

  // Set up zoom support
  let svg = d3.select("svg.static-repo"),
  inner = svg.append("g"),
  zoom = d3.zoom().on("zoom", function() {
    inner.attr("transform", d3.event.transform);
  });
  svg.call(zoom);

  let render = new dagreD3.render();

  // Left-to-right layout
  let g = new dagreD3.graphlib.Graph()
  .setGraph({
    nodesep: 100,
    ranksep: 70,
    rankdir: "LR"
  }),

  repo = await fetchJSON(path);

  async function getTotalNumberOfPipelines(r){
    await r;

    let n = 1;
    for (let component in r){
      if (r[component]["abstract"]){
        n = n * r[component]["concreteImplementations"].length;
      }
      else if (r[component]["configIndexes"]){
        n = n * r[component]["configIndexes"].length
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

  await draw(repo, g, svg, zoom, inner, render);
}


async function dynamicGraph(path="dynamic_repo.json") {

  let svg = d3.select("svg.dynamic-repo"),
  inner = svg.append("g"),
  zoom = d3.zoom().on("zoom", function() {
    inner.attr("transform", d3.event.transform);
  });
  svg.call(zoom);

  let render = new dagreD3.render();

  // Left-to-right layout
  let g = new dagreD3.graphlib.Graph()
  .setGraph({
    nodesep: 100,
    ranksep: 70,
    rankdir: "LR"
  });

  let repo = await fetchJSON(path);
  await draw(repo, g, svg, zoom, inner, render, true);

  // status updating commands
  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  const n_tasks = Object.keys(repo).length;
  let n_done = 0;

  while (n_tasks >= n_done){
    let repo  =await fetchJSON(path);
    for (const k in repo){
      if (repo[k]["status"] === "DONE"){
        n_done +=1;
      } else {
        n_done =0;
      }
    }
    async function update(){
      for (let task in repo) {
        let element = d3.select("#" + task);
        if (element.attr("class") !== "node " + repo[task]["status"]) {
          element.attr("class", "node " + repo[task]["status"])
        }
      }
    }
    await update();
    await sleep(1000);
  }
}

document.addEventListener("DOMContentLoaded", staticGraph());
document.addEventListener("DOMContentLoaded", dynamicGraph());
