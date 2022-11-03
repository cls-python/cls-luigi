let config = 'config.json';

async function fetchJSON(path){
  const fetched = await fetch(path);

  return fetched.json();
}

async function combineRawPipelinesToOne(r){
  await r;
  let newObj = {};
  for (let k in r){
    newObj = Object.assign({}, newObj, r[k]);
  }
  return newObj;
}

async function addNodesAndEdges(JSONPipelines, graph, static=true) {
  await JSONPipelines;

  for (let component in JSONPipelines) {
    let componentDetails = JSONPipelines[component];
    let html = "<div>";
    let className;


    if (static === true){
      html += "<span class=status></span>";
      html += "<span class=name>" + component + "</span>";
      html += "<span class=queue>" + "" + "</span>";

      if (componentDetails.concreteImplementations) {
        className = "abstractComponent";
        componentDetails.concreteImplementations.map(function(item){
          html += "<span class=queue>" + "  " + item + "</span>";
        })
        html += "<span class=queue>" + "" + "</span>";
      }
      else if (componentDetails.configIndexes){
        className = "indexedComponent";
        let indexes = Object.keys(componentDetails.configIndexes);

        indexes.map(function(i) {
          componentDetails.configIndexes[i].map(function(j) {
            html += "<span class=queue>" + i + " : " + j + "</span>";
          })
        })
        html += "<span class=queue>" + "" + "</span>";
      }
      else {
        className = "notAbstractComponent";
      }
    }

    else if (static === false) {
      className = componentDetails["status"];
      if (className === "RUNNING"){
        className += " warn";
      }
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

      componentDetails.inputQueue.map(function(d) {
        graph.setEdge(d, component, {});
      })
    }
  }
}

async function draw(JSONPipelines, g, svg, zoom, inner, render, static=true){

  await addNodesAndEdges(JSONPipelines, g,  static);

  inner.call(render, g);

  // Zoom and scale to fit
  let graphWidth =  g.graph().width + 80;
  let graphHeight =  g.graph().height + 200;
  let width =  parseInt(svg.style("width").replace(/px/, ""));
  let height =  parseInt(svg.style("height").replace(/px/, ""));
  let zoomScale =  Math.min(width / graphWidth, height / graphHeight);
  let translateX =  (width / 2) - ((graphWidth * zoomScale) / 2) + 25;
  let translateY =  (height / 2) - ((graphHeight * zoomScale) / 2) + 10;
  svg.call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(zoomScale));

}


async function staticGraph() {

  let path = await fetchJSON(config);
  path = path['static_pipeline']

  // Set up zoom support
  let svg = d3.select("svg.static-pipeline"),
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

  JSONPipeline = await fetchJSON(path);

  async function getTotalNumberOfPipelines(r){
    await r;

    let n = 1;

    for (let component in r){
      if (r[component]["abstract"]){
        n = n * r[component]["concreteImplementations"].length;
      }
      else if (r[component]["configIndexes"]){
        n = n * Object.keys(r[component]["configIndexes"]).length
      }
    }
    return n;
  }

  d3.select("total-pipelines")
    .append("div")
    .classed("total-pipelines", true)
    .append("div")
    .classed("title", true)
    .text("Total Number of Pipelines: " + await getTotalNumberOfPipelines(JSONPipeline));

  await draw(JSONPipeline, g, svg, zoom, inner, render);
}

async function dynamicGraph() {

   let path = await fetchJSON(config);
   path = path['dynamic_pipeline']


  let svg = d3.select("svg.dynamic-pipeline"),
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

  let rawPipelinesJSON = await fetchJSON(path);
  let combinedPipeline = await combineRawPipelinesToOne(rawPipelinesJSON);


  async function getTotalNumberOfTasks(r){
    await r;
    return Object.keys(r).length;
  }

  d3.select("total-tasks")
    .append("div")
    .classed("total-tasks", true)
    .append("div")
    .classed("title", true)
    .text("Total Number \nof Tasks: " + await getTotalNumberOfTasks(combinedPipeline));



  await draw(combinedPipeline, g, svg, zoom, inner, render, false);

  // status updating commands
  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  const n_tasks = Object.keys(combinedPipeline).length;
  let n_done = 0;

  while (n_tasks >= n_done){
    let rawPipelines  =await fetchJSON(path);
    let combinedPipeline = await combineRawPipelinesToOne(rawPipelines);


    for (const k in combinedPipeline){
      if (combinedPipeline[k]["status"] === "DONE"){
        n_done +=1;
      } else {
        n_done =0;
      }
    }
    async function update(){
      for (let task in combinedPipeline) {
        let element = d3.select("#" + task);
        let node_status = combinedPipeline[task]["status"]
        if (node_status == "RUNNING"){
          node_status += " warn";
        }
        if (element.attr("class") !== "node " + node_status) {
          element.attr("class", "node " + node_status)
        }
      }
    }
    await update();
    await sleep(1500);
  }
}

document.addEventListener("DOMContentLoaded", staticGraph());
document.addEventListener("DOMContentLoaded", dynamicGraph());
