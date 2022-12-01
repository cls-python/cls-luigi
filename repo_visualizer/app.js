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

async function draw(JSONPipelines, g, svg, zoom, inner, render, static=true, controlYPosition=25){

  await addNodesAndEdges(JSONPipelines, g,  static);

  inner.call(render, g);

  // Zoom and scale to fit
  let graphWidth =  g.graph().width + 80;
  let graphHeight =  g.graph().height + 200;
  let width =  parseInt(svg.style("width").replace(/px/, ""));
  let height =  parseInt(svg.style("height").replace(/px/, ""));
  let zoomScale =  Math.min(width / graphWidth, height / graphHeight);
  let translateX =  (width / 2) - ((graphWidth * zoomScale) / 2) + 25;
  let translateY =  (height / 2) - ((graphHeight * zoomScale) /2) + controlYPosition ;
  svg.call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(zoomScale));

}

async function initRenderAndGraph(graphNodeSep=300, graphRankSep=70, rankDir="LR"){
  let render = new dagreD3.render();

  // Left-to-right layout
  let g = new dagreD3.graphlib.Graph()
  .setGraph({
    nodesep: graphNodeSep,
    ranksep: graphRankSep,
    rankdir: rankDir
  });

  return [render, g]


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

  let renderAndGraph = await initRenderAndGraph()
  const render = renderAndGraph[0],
        g      = renderAndGraph[1];

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


  await draw(JSONPipeline, g, svg, zoom, inner, render, true, -25);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
async function updateTaskStatus(pipeline, div){
  await pipeline
  for (let task in pipeline) {
    let element = d3.select(div).select("#" + task);
    let node_status = pipeline[task]["status"];
    if (node_status == "RUNNING"){
      node_status += " warn";
    }
    if (element.attr("class") !== "node " + node_status) {
      element.attr("class", "node " + node_status)
    }
  }
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

  let renderAndGraph = await initRenderAndGraph(graphNodeSep=100)
  const render = renderAndGraph[0],
        g      = renderAndGraph[1];


  let rawPipelinesJSON = await fetchJSON(path);
  let combinedPipeline = await combineRawPipelinesToOne(rawPipelinesJSON);


  async function getTotalNumberOfTasks(r){
    await r;
    return Object.keys(r).length;
  }

  d3.select("#dynamic_p")
    .append("div")
    .classed("title", true)
    .text("Unique Tasks: " + await getTotalNumberOfTasks(combinedPipeline));



  await draw(combinedPipeline, g, svg, zoom, inner, render, false, -200);

  const n_tasks = Object.keys(combinedPipeline).length;
  let n_done = 0;

  while (n_tasks !== n_done){
    let rawPipelines  =await fetchJSON(path);
    let combinedPipeline = await combineRawPipelinesToOne(rawPipelines);


    for (const k in combinedPipeline) {
      if (combinedPipeline[k]["status"] === "DONE") {
        n_done += 1;
      }
    }
    await updateTaskStatus(combinedPipeline, ".dynamic-pipeline");
    await sleep(1500);

      if (n_tasks !== n_done){
        n_done = 0;
      }
      else if (n_tasks === n_done){
        let startTimes = [];
        let lastupdatedTimes = [];


        for (const k in combinedPipeline){
          startTimes.push(combinedPipeline[k]["timeRunning"]);
          lastupdatedTimes.push(combinedPipeline[k]["lastUpdated"]);
        }

        let start = Math.min.apply(null, startTimes)
        let end = Math.max.apply(null, lastupdatedTimes)
        const total = ((end - start)  / 60).toFixed(2)
        d3.select("#dynamic_p")
          .append("div")
          .classed("title", true)
          .style("text-align", "left")
          .style("padding-top", "20px")
          .text("Executed in: " + total + " min");


      }
    }
}

async function removeOldGraphAndDrawNew(selectedPipeline, svg){
  let oldGraph = svg.selectAll("g");
  if (oldGraph.empty() === false){
    svg.selectAll("g").remove();
  }
  let renderAndGraph = await initRenderAndGraph(graphNodeSep=300)
  const render = renderAndGraph[0],
        g      = renderAndGraph[1];

  let inner = svg.append("g"),
    zoom = d3.zoom().on("zoom", function() {
    inner.attr("transform", d3.event.transform);
  });
  svg.call(zoom);

  await draw(selectedPipeline, g, svg, zoom, inner, render, false, -350);
}

async function singlePipelines(){

  let path = await fetchJSON(config);
  path = path['dynamic_pipeline']
  let rawPipelinesJSON = await fetchJSON(path);
  let indecies = Object.keys(rawPipelinesJSON);
  indecies.unshift("Pipeline Index"); // default hidden val in dropdown


  let svg = d3.select("svg.single-pipeline");
  svg.style("margin-top", "-320px");

  // add the options to the button
  d3.select("#selectButton")
    .selectAll('myOptions')
      .data(indecies)
    .enter()
      .append('option')
      .text(" ")
    .text(function (d) { return d; })
    .attr("value", function (d) { return d; })
      .each(function (d) {
        if (d === "Pipeline Index") {d3.select(this).property("disabled", true)}
  })


  d3.select('#selectButton')
      .on("change", async function(d) {

        var selectedIndex = d3.select(this).property("value");

        let rawPipelinesJSON = await fetchJSON(path);
        let selectedPipeline = rawPipelinesJSON[selectedIndex];

        removeOldGraphAndDrawNew(selectedPipeline, svg);

        async function getTotalNumberOfTasks(r){
          await r;
          return Object.keys(r).length;
        }
        let old_n_tasks = d3.select("#single_p_n_tasks").selectAll("div.legend-title");
        if (old_n_tasks.empty() === false){
          old_n_tasks.remove();
        }


        d3.select("#single_p_n_tasks")
            .append("div")
            .classed("legend-title", true)
            .style("text-align", "left")
            .text("N Tasks: " + await getTotalNumberOfTasks(selectedPipeline));


        const n_tasks = Object.keys(selectedPipeline).length;
        let n_done = 0;
        while (n_tasks !== n_done){
          let rawPipelinesJSON = await fetchJSON(path);
          let pipeline = rawPipelinesJSON[selectedIndex];
          let parsedIntSelectedIndex = parseInt(selectedIndex);

          for (const k in pipeline){
            if (pipeline[k]["status"] === "DONE"){
              n_done +=1;
            }
          }
          await updateTaskStatus(pipeline, ".single-pipeline");
          await sleep(1500);

          if (n_tasks !== n_done){
            n_done = 0;
          }
          else if (n_tasks === n_done){
            let startTimes = [];
            let lastupdatedTimes = [];

            let start;
            let end;


            if (parsedIntSelectedIndex  === 0){
              for (const k in pipeline){
                startTimes.push(pipeline[k]["timeRunning"]);
                lastupdatedTimes.push(pipeline[k]["lastUpdated"]);
              }
              start = Math.min.apply(null , startTimes);
              end = Math.max.apply(null, lastupdatedTimes);
            }
            else if  (parsedIntSelectedIndex !== 0){
              let previousIndex = (parsedIntSelectedIndex - 1).toString();
              for (const k in rawPipelinesJSON[previousIndex]){
                startTimes.push(rawPipelinesJSON[previousIndex][k]["lastUpdated"]);
              }
              for (const k in pipeline){
                lastupdatedTimes.push(pipeline[k]["lastUpdated"]);
              }
              start = Math.max.apply(null , startTimes);
              end = Math.max.apply(null, lastupdatedTimes);
            }

            const total = ((end - start)  / 60).toFixed(2)


            d3.select("#single_p_n_tasks")
              .append("div")
              .classed("legend-title", true)
              .style("text-align", "left")
              .style("padding-top", "20px")
              .text("Executed in: " + total + " min");
          }
        }
      })
}

document.addEventListener("DOMContentLoaded", staticGraph());
document.addEventListener("DOMContentLoaded", dynamicGraph());
document.addEventListener("DOMContentLoaded", singlePipelines());
